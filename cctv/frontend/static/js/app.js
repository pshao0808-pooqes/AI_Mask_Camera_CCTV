const socket = io();

class TrafficMonitor {
    constructor() {
        this.socket = io();
        this.map = null;
        this.cctvList = [];
        this.activeStreams = new Set();
        this.trafficData = {};
        this.markers = {};
        this.deviceInfo = null;
        this.accidentMarkers = [];
        this.showAccidents = false;

        
        this.initializeSocket();
        this.initializeMap();
        this.loadCCTVList();
        this.updateStats();
        this.initializeMacOSFeatures();
        
    }

    initializeMacOSFeatures() {
        // macOS 스타일 스크롤 부드럽게
        document.documentElement.style.scrollBehavior = 'smooth';
        
        // 키보드 네비게이션 지원
        this.setupKeyboardNavigation();
        
        // 다크모드 감지
        this.setupDarkModeDetection();
    }

    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            if (e.metaKey || e.ctrlKey) {
                switch(e.key) {
                    case '1':
                        e.preventDefault();
                        this.focusFirstCCTV();
                        break;
                    case 'a':
                        e.preventDefault();
                        this.toggleAllCCTV();
                        break;
                }
            }
        });
    }

    setupDarkModeDetection() {
        const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
        darkModeQuery.addEventListener('change', (e) => {
            this.updateMapTheme(e.matches);
        });
    }

    updateMapTheme(isDark) {
        if (this.map) {
            // 다크모드에 따른 지도 스타일 변경
            const tileLayer = isDark ? 
                'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' :
                'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
            
            this.map.eachLayer((layer) => {
                if (layer instanceof L.TileLayer) {
                    this.map.removeLayer(layer);
                }
            });
            
            L.tileLayer(tileLayer, {
                attribution: '© OpenStreetMap contributors'
            }).addTo(this.map);
        }
    }

    initializeSocket() {
        this.socket.on('connect', () => {
            console.log('서버에 연결되었습니다.');
            this.updateConnectionStatus(true);
        });

        this.socket.on('disconnect', () => {
            console.log('서버 연결이 끊어졌습니다.');
            this.updateConnectionStatus(false);
        });

        this.socket.on('device_info', (data) => {
            this.deviceInfo = data;
            this.updateDeviceStatus(data);
        });

        this.socket.on('cctv_list', (data) => {
            this.cctvList = data;
            this.renderCCTVList();
            this.addMarkersToMap();
        });

        this.socket.on('traffic_update', (data) => {
            this.trafficData[data.cctv_id] = data.data;
            this.updateMarker(data.cctv_id, data.data);
            this.updateCCTVItem(data.cctv_id, data.data);
            this.updateStats();
        });

        this.socket.on('frame_update', (data) => {
            this.updateVideoFrame(data.cctv_id, data.frame);
        });
    }

    updateConnectionStatus(isConnected) {
        const statusEl = document.getElementById('connection-status');
        const statusDot = document.createElement('span');
        statusDot.className = `status-dot ${isConnected ? 'connected' : 'disconnected'}`;
        
        statusEl.innerHTML = '';
        statusEl.appendChild(statusDot);
        statusEl.appendChild(document.createTextNode(isConnected ? '연결됨' : '연결 끊어짐'));
        statusEl.className = isConnected ? 'status-connected' : 'status-disconnected';
    }

    updateDeviceStatus(deviceInfo) {
        const deviceStatusEl = document.getElementById('device-status');
        if (deviceStatusEl) {
            const deviceType = deviceInfo.device === 'cuda' ? 'GPU' : 'CPU';
            const deviceClass = deviceInfo.device === 'cuda' ? 'device-gpu' : 'device-cpu';
            const icon = deviceInfo.device === 'cuda' ? '⚡' : '🖥️';
            
            deviceStatusEl.innerHTML = `${icon} ${deviceType} 가속`;
            deviceStatusEl.className = `device-status ${deviceClass}`;
        }

        this.updateSystemInfo(deviceInfo);
    }

    updateSystemInfo(deviceInfo) {
        const systemInfoEl = document.getElementById('system-info-details');
        if (systemInfoEl) {
            let detailsHtml = `<strong>처리 장치:</strong> ${deviceInfo.type}<br>`;
            detailsHtml += `<strong>모델:</strong> ${deviceInfo.name}<br>`;
            
            if (deviceInfo.details) {
                Object.entries(deviceInfo.details).forEach(([key, value]) => {
                    const labels = {
                        'threads': '스레드 수',
                        'total_cores': '총 코어 수',
                        'memory_total': '총 메모리',
                        'cuda_version': 'CUDA 버전',
                        'device_count': 'GPU 개수'
                    };
                    const label = labels[key] || key;
                    detailsHtml += `<strong>${label}:</strong> ${value}<br>`;
                });
            }
            
            if (deviceInfo.runtime) {
                detailsHtml += '<br><strong>실시간 정보:</strong><br>';
                Object.entries(deviceInfo.runtime).forEach(([key, value]) => {
                    const labels = {
                        'memory_allocated': '할당된 메모리',
                        'memory_cached': '캐시된 메모리',
                        'utilization': 'GPU 사용률'
                    };
                    const label = labels[key] || key;
                    detailsHtml += `<strong>${label}:</strong> ${value}<br>`;
                });
            }
            
            systemInfoEl.innerHTML = detailsHtml;
        }
    }

    initializeMap() {
        this.map = L.map('map').setView([37.5665, 126.9780], 11);
        
        // 다크모드 감지하여 초기 타일 설정
        const isDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
        const tileLayer = isDarkMode ? 
            'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' :
            'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
        
        L.tileLayer(tileLayer, {
            attribution: '© OpenStreetMap contributors'
        }).addTo(this.map);
        this.setupAccidentToggle();
    }

    loadCCTVList() {
        fetch('/api/cctv/list')
            .then(response => response.json())
            .then(data => {
                this.cctvList = data;
                this.renderCCTVList();
                this.addMarkersToMap();
            })
            .catch(error => {
                console.error('CCTV 목록 로드 실패:', error);
            });
    }

    renderCCTVList() {
        const cctvListElement = document.getElementById('cctv-list');
        
        if (this.cctvList.length === 0) {
            cctvListElement.innerHTML = '<div class="loading">CCTV를 찾을 수 없습니다.</div>';
            return;
        }

        cctvListElement.innerHTML = this.cctvList.map(cctv => {
            const trafficInfo = this.trafficData[cctv.id] || {};
            const isActive = this.activeStreams.has(cctv.id);
            
            // 트래픽 레벨에 따른 클래스 설정
            let trafficClass = '';
            if (trafficInfo.traffic_level === '원활') trafficClass = 'smooth';
            else if (trafficInfo.traffic_level === '보통') trafficClass = 'normal';
            else if (trafficInfo.traffic_level === '혼잡') trafficClass = 'congested';
            
            return `
                <div class="cctv-item ${isActive ? 'active' : ''}" 
                     data-cctv-id="${cctv.id}" 
                     id="cctv-item-${cctv.id}"
                     tabindex="0">
                    <button class="dashboard-button" onclick="trafficMonitor.openDashboard(${cctv.id})" title="대시보드 열기">
                        📊
                    </button>
                    <h4>${cctv.name}</h4>
                    <div class="cctv-info">
                        ${cctv.lat.toFixed(4)}, ${cctv.lon.toFixed(4)}
                    </div>
                    ${trafficInfo.traffic_level ? `
                        <div class="traffic-info">
                            <span class="traffic-level ${trafficClass}">
                                ${trafficInfo.traffic_level}
                            </span>
                            <div class="traffic-detail">차량 ${trafficInfo.vehicle_count}대</div>
                            <div class="traffic-detail">${trafficInfo.avg_speed.toFixed(1)}km/h</div>
                            <div class="device-indicator device-${trafficInfo.device.toLowerCase()}">
                                ${trafficInfo.device} 처리
                            </div>
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');

        // 클릭 및 키보드 이벤트 추가
        cctvListElement.querySelectorAll('.cctv-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // 대시보드 버튼 클릭이 아닌 경우에만 토글
                if (!e.target.classList.contains('dashboard-button')) {
                    const cctvId = parseInt(item.dataset.cctvId);
                    this.toggleCCTVMonitoring(cctvId);
                }
            });
            
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    const cctvId = parseInt(item.dataset.cctvId);
                    this.toggleCCTVMonitoring(cctvId);
                }
            });
        });
    }

    openDashboard(cctvId) {
        const cctv = this.cctvList.find(c => c.id === cctvId);
        if (cctv) {
            const dashboardUrl = `/dashboard/${cctvId}`;
            const dashboardWindow = window.open(
                dashboardUrl, 
                `dashboard_${cctvId}`,
                'width=1400,height=900,scrollbars=yes,resizable=yes'
            );
            
            if (dashboardWindow) {
                dashboardWindow.focus();
            }
        }
    }

    updateCCTVItem(cctvId, trafficData) {
        const cctvItem = document.getElementById(`cctv-item-${cctvId}`);
        if (cctvItem) {
            let trafficClass = '';
            if (trafficData.traffic_level === '원활') trafficClass = 'smooth';
            else if (trafficData.traffic_level === '보통') trafficClass = 'normal';
            else if (trafficData.traffic_level === '혼잡') trafficClass = 'congested';
            
            const trafficInfoEl = cctvItem.querySelector('.traffic-info');
            const trafficInfoHtml = `
                <div class="traffic-info">
                    <span class="traffic-level ${trafficClass}">
                        ${trafficData.traffic_level}
                    </span>
                    <div class="traffic-detail">차량 ${trafficData.vehicle_count}대</div>
                    <div class="traffic-detail">${trafficData.avg_speed.toFixed(1)}km/h</div>
                    <div class="device-indicator device-${trafficData.device.toLowerCase()}">
                        ${trafficData.device} 처리
                    </div>
                </div>
            `;
            
            if (trafficInfoEl) {
                trafficInfoEl.outerHTML = trafficInfoHtml;
            } else {
                cctvItem.insertAdjacentHTML('beforeend', trafficInfoHtml);
            }
        }
    }

    addMarkersToMap() {
        this.cctvList.forEach(cctv => {
            const trafficInfo = this.trafficData[cctv.id] || {};
            
            // macOS 스타일 컬러 적용
            let color = '#8E8E93'; // 기본 시스템 그레이
            if (trafficInfo.traffic_level === '원활') color = '#34C759'; // 시스템 그린
            else if (trafficInfo.traffic_level === '보통') color = '#FF9500'; // 시스템 오렌지
            else if (trafficInfo.traffic_level === '혼잡') color = '#FF3B30'; // 시스템 레드
            
            const marker = L.circleMarker([cctv.lat, cctv.lon], {
                radius: 8,
                fillColor: color,
                color: '#FFFFFF',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);

            const popupContent = this.createPopupContent(cctv, trafficInfo);
            marker.bindPopup(popupContent);
            this.markers[cctv.id] = marker;
        });
    }

    createPopupContent(cctv, trafficInfo) {
        const isActive = this.activeStreams.has(cctv.id);
        const buttonText = isActive ? '모니터링 중지' : '모니터링 시작';
        const buttonClass = isActive ? 'secondary' : '';
        
        return `
            <div style="min-width: 220px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">
                <h4 style="margin-bottom: 8px; font-size: 15px; font-weight: 600;">${cctv.name}</h4>
                <p style="margin-bottom: 6px; font-size: 12px; color: #8E8E93;">
                    <strong>위치:</strong> ${cctv.lat.toFixed(4)}, ${cctv.lon.toFixed(4)}
                </p>
                ${trafficInfo.traffic_level ? `
                    <p style="margin-bottom: 4px; font-size: 12px;"><strong>트래픽:</strong> ${trafficInfo.traffic_level}</p>
                    <p style="margin-bottom: 4px; font-size: 12px;"><strong>차량 수:</strong> ${trafficInfo.vehicle_count}대</p>
                    <p style="margin-bottom: 4px; font-size: 12px;"><strong>평균 속도:</strong> ${trafficInfo.avg_speed.toFixed(1)}km/h</p>
                    <p style="margin-bottom: 4px; font-size: 12px;"><strong>처리 장치:</strong> ${trafficInfo.device}</p>
                    <p style="margin-bottom: 8px; font-size: 11px; color: #8E8E93;">
                        ${new Date(trafficInfo.timestamp).toLocaleTimeString()}
                    </p>
                ` : '<p style="margin-bottom: 8px; font-size: 12px; color: #8E8E93;">트래픽 정보 없음</p>'}
                <div style="display: flex; gap: 8px;">
                    <button onclick="trafficMonitor.toggleCCTVMonitoring(${cctv.id})" 
                            class="mac-button ${buttonClass}"
                            style="flex: 1;">
                        ${buttonText}
                    </button>
                    <button onclick="trafficMonitor.openDashboard(${cctv.id})" 
                            class="mac-button"
                            style="padding: 8px;">
                        📊
                    </button>
                </div>
            </div>
        `;
    }

    updateMarker(cctvId, trafficData) {
        const marker = this.markers[cctvId];
        if (marker) {
            let color = '#8E8E93';
            if (trafficData.traffic_level === '원활') color = '#34C759';
            else if (trafficData.traffic_level === '보통') color = '#FF9500';
            else if (trafficData.traffic_level === '혼잡') color = '#FF3B30';
            
            marker.setStyle({ fillColor: color });
            
            const cctv = this.cctvList.find(c => c.id === cctvId);
            if (cctv) {
                const popupContent = this.createPopupContent(cctv, trafficData);
                marker.setPopupContent(popupContent);
            }
        }
    }

    toggleCCTVMonitoring(cctvId) {
        if (this.activeStreams.has(cctvId)) {
            this.stopCCTVMonitoring(cctvId);
        } else {
            this.startCCTVMonitoring(cctvId);
        }
    }

    startCCTVMonitoring(cctvId) {
        fetch(`/api/cctv/${cctvId}/start`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(`CCTV ${cctvId} 모니터링 시작:`, data);
                this.activeStreams.add(cctvId);
                this.addVideoStream(cctvId);
                this.renderCCTVList();
                this.updateVideoGrid();
            })
            .catch(error => {
                console.error(`CCTV ${cctvId} 모니터링 시작 실패:`, error);
            });
    }

    stopCCTVMonitoring(cctvId) {
        fetch(`/api/cctv/${cctvId}/stop`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log(`CCTV ${cctvId} 모니터링 중지:`, data);
                this.activeStreams.delete(cctvId);
                this.removeVideoStream(cctvId);
                this.renderCCTVList();
                this.updateVideoGrid();
            })
            .catch(error => {
                console.error(`CCTV ${cctvId} 모니터링 중지 실패:`, error);
            });
    }

    addVideoStream(cctvId) {
        const cctv = this.cctvList.find(c => c.id === cctvId);
        if (!cctv) return;

        const videoGrid = document.getElementById('video-grid');
        
        const noVideo = videoGrid.querySelector('.no-video');
        if (noVideo) {
            noVideo.remove();
        }

        const videoItem = document.createElement('div');
        videoItem.className = 'video-item';
        videoItem.id = `video-${cctvId}`;
        videoItem.innerHTML = `
            <img id="frame-${cctvId}" src="" alt="CCTV ${cctvId}" style="display: none;">
            <div class="video-overlay">
                <div class="video-title">${cctv.name}</div>
                <div class="video-stats" id="stats-${cctvId}">연결 중...</div>
            </div>
        `;

        videoGrid.appendChild(videoItem);
    }

    removeVideoStream(cctvId) {
        const videoItem = document.getElementById(`video-${cctvId}`);
        if (videoItem) {
            videoItem.remove();
        }
        this.updateVideoGrid();
    }

    updateVideoFrame(cctvId, frameBase64) {
        const frameImg = document.getElementById(`frame-${cctvId}`);
        if (frameImg) {
            frameImg.src = `data:image/jpeg;base64,${frameBase64}`;
            frameImg.style.display = 'block';
        }

        const trafficInfo = this.trafficData[cctvId];
        if (trafficInfo) {
            const statsElement = document.getElementById(`stats-${cctvId}`);
            if (statsElement) {
                statsElement.innerHTML = `${trafficInfo.traffic_level} • ${trafficInfo.vehicle_count}대`;
            }
        }
    }

    updateVideoGrid() {
        const videoGrid = document.getElementById('video-grid');
        const videoItems = videoGrid.querySelectorAll('.video-item');
        
        if (videoItems.length === 0) {
            videoGrid.innerHTML = `
                <div class="no-video">
                    <i class="fas fa-video-slash"></i>
                    <p>모니터링 중인 CCTV가 없습니다</p>
                    <p>좌측 목록에서 CCTV를 선택해주세요</p>
                </div>
            `;
        }
    }

    updateStats() {
        const activeCctvCount = this.activeStreams.size;
        const totalVehicles = Object.values(this.trafficData)
            .reduce((sum, data) => sum + (data.vehicle_count || 0), 0);
        const avgSpeed = Object.values(this.trafficData).length > 0 ?
            Object.values(this.trafficData)
                .reduce((sum, data, _, arr) => sum + (data.avg_speed || 0) / arr.length, 0) : 0;

        document.getElementById('active-cctv-count').textContent = activeCctvCount;
        document.getElementById('total-vehicles').textContent = totalVehicles.toLocaleString();
        document.getElementById('avg-speed').textContent = `${avgSpeed.toFixed(1)} km/h`;
    }

    // 키보드 네비게이션 헬퍼 메서드
    focusFirstCCTV() {
        const firstCCTV = document.querySelector('.cctv-item');
        if (firstCCTV) {
            firstCCTV.focus();
        }
    }

    toggleAllCCTV() {
        if (this.activeStreams.size > 0) {
            // 모든 CCTV 중지
            Array.from(this.activeStreams).forEach(cctvId => {
                this.stopCCTVMonitoring(cctvId);
            });
        } else {
            // 처음 3개 CCTV 시작
            this.cctvList.slice(0, 3).forEach(cctv => {
                this.startCCTVMonitoring(cctv.id);
            });
        }
    }
    setupAccidentToggle() {
        const mapContainer = document.querySelector('.map-container');
        const toggleHtml = `
            <button id="accident-toggle" class="mac-button secondary" 
                    style="position: absolute; top: 60px; right: 20px; z-index: 1000;">
                🚨 사고지점
            </button>
        `;
        mapContainer.insertAdjacentHTML('beforeend', toggleHtml);
        
        document.getElementById('accident-toggle').addEventListener('click', () => {
            this.toggleAccidents();
        });
    }

    async toggleAccidents() {
        const btn = document.getElementById('accident-toggle');
        
        if (!this.showAccidents) {
            btn.textContent = '로딩중...';
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/accidents/data');
                const result = await response.json();
                
                if (result.success) {
                    this.showAccidentMarkers(result.data);
                    this.showAccidents = true;
                    btn.textContent = '❌ 사고지점 숨김';
                    btn.classList.remove('secondary');
                }
            } catch (error) {
                console.error('사고 데이터 로딩 실패:', error);
                btn.textContent = '🚨 사고지점';
            }
            
            btn.disabled = false;
        } else {
            this.hideAccidentMarkers();
            this.showAccidents = false;
            btn.textContent = '🚨 사고지점';
            btn.classList.add('secondary');
        }
    }

    showAccidentMarkers(accidents) {
        accidents.forEach(accident => {
            let color = '#FFCC00'; // 기본 노란색
            let size = 8;
            
            if (accident.severity === '사망') {
                color = '#FF3B30'; // 빨간색
                size = 12;
            } else if (accident.severity === '중상') {
                color = '#FF9500'; // 주황색
                size = 10;
            }
            
            const marker = L.circleMarker([accident.lat, accident.lon], {
                radius: size,
                fillColor: color,
                color: '#FFFFFF',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(this.map);
            
            const popupContent = `
                <div style="min-width: 200px;">
                    <h4>${accident.spot_name}</h4>
                    <p><strong>지역:</strong> ${accident.address}</p>
                    <p><strong>사고건수:</strong> ${accident.accident_count}건</p>
                    <p><strong>사망자:</strong> ${accident.death_count}명</p>
                    <p><strong>부상자:</strong> ${accident.injury_count}명</p>
                    <p><strong>심각도:</strong> ${accident.severity}</p>
                </div>
            `;
            
            marker.bindPopup(popupContent);
            this.accidentMarkers.push(marker);
        });
        
        console.log(`${accidents.length}개의 사고지점이 표시되었습니다.`);
    }

    hideAccidentMarkers() {
        this.accidentMarkers.forEach(marker => {
            this.map.removeLayer(marker);
        });
        this.accidentMarkers = [];
    }
}

// 전역 변수로 설정
let trafficMonitor;

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    trafficMonitor = new TrafficMonitor();
});
