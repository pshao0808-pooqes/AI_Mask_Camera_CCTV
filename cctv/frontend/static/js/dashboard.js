class SimpleDashboard {
    constructor(cctvId, cctvInfo) {
        this.cctvId = cctvId;
        this.cctvInfo = cctvInfo;
        this.socket = io();
        this.isMonitoring = false;
        this.trafficData = {
            totalIn: 0,
            totalOut: 0,
            hourlyData: []
        };

        // 차트 추가
        this.chart = null;
        this.chartData = {
            labels: [],
            entered: [],
            exited: [],
            current: []
        };
        
        this.initializeSocket();
        this.setupControls();
        this.addChart(); // 차트 추가
        this.generateInitialData();
        this.updateFlowTable();
    }

    initializeSocket() {
        this.socket.on('connect', () => {
            console.log('대시보드 연결됨');
            this.socket.emit('join_dashboard', { cctv_id: this.cctvId });
        });

        this.socket.on('dashboard_frame_update', (data) => {
            if (data.cctv_id === this.cctvId) {
                this.updateVideoFrame(data.frame);
                this.updateRealtimeInfo(data.data);
            }
        });

        this.socket.on('traffic_update', (data) => {
            if (data.cctv_id === this.cctvId) {
                console.log('받은 트래픽 데이터:', data.data); // 디버깅용
                this.updateTrafficData(data.data);
            }
        });
    }

    setupControls() {
        // 시작/중지 버튼
        document.getElementById('start-monitoring')?.addEventListener('click', () => {
            this.startMonitoring();
        });

        document.getElementById('stop-monitoring')?.addEventListener('click', () => {
            this.stopMonitoring();
        });

        // 리셋 버튼
        document.getElementById('reset-data')?.addEventListener('click', () => {
            this.resetData();
        });

        // 내보내기 버튼
        document.getElementById('export-data')?.addEventListener('click', () => {
            this.exportData();
        });

        // 민감도 슬라이더
        const sensitivitySlider = document.getElementById('sensitivity-slider');
        const sensitivityValue = document.getElementById('sensitivity-value');
        
        sensitivitySlider?.addEventListener('input', (e) => {
            sensitivityValue.textContent = e.target.value;
        });
    }

    addChart() {
        const dataSection = document.querySelector('.data-section');
        if (dataSection) {
            const chartHTML = `
                <div class="section-card">
                    <h3><i class="fas fa-chart-line"></i> 실시간 트래픽</h3>
                    <canvas id="traffic-chart" width="400" height="200" style="max-height: 300px;"></canvas>
                </div>
            `;
            dataSection.insertAdjacentHTML('afterbegin', chartHTML);
            
            // Chart.js 로드 대기 후 차트 생성
            this.waitForChart().then(() => {
                this.createChart();
            });
        }
    }

    async waitForChart() {
        return new Promise((resolve) => {
            if (typeof Chart !== 'undefined') {
                resolve();
            } else {
                const checkChart = setInterval(() => {
                    if (typeof Chart !== 'undefined') {
                        clearInterval(checkChart);
                        resolve();
                    }
                }, 100);
            }
        });
    }

    createChart() {
        const ctx = document.getElementById('traffic-chart');
        if (ctx && typeof Chart !== 'undefined') {
            console.log('차트 생성 시작'); // 디버깅용
            this.chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: '진입',
                        data: [],
                        borderColor: '#4CAF50',
                        backgroundColor: 'rgba(76, 175, 80, 0.1)',
                        fill: true
                    }, {
                        label: '진출', 
                        data: [],
                        borderColor: '#2196F3',
                        backgroundColor: 'rgba(33, 150, 243, 0.1)',
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
            console.log('차트 생성 완료'); // 디버깅용
        } else {
            console.log('차트 생성 실패 - Chart.js 미로드 또는 캔버스 없음'); // 디버깅용
        }
    }

    generateInitialData() {
        this.trafficData.hourlyData = [];
        const now = new Date();
        
        // 최근 12시간 데이터 생성
        for (let i = 11; i >= 0; i--) {
            const time = new Date(now.getTime() - (i * 60 * 60 * 1000));
            const hour = time.getHours();
            
            // 시간대별 트래픽 패턴 시뮬레이션
            const baseTraffic = this.getBaseTrafficForHour(hour);
            const inCount = Math.floor(Math.random() * 5) + baseTraffic;
            const outCount = Math.floor(Math.random() * 5) + baseTraffic;
            
            this.trafficData.hourlyData.push({
                time: time.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
                in: inCount,
                out: outCount
            });
            
            this.trafficData.totalIn += inCount;
            this.trafficData.totalOut += outCount;
        }
    }

    getBaseTrafficForHour(hour) {
        // 출퇴근 시간대 트래픽 증가
        if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) {
            return 8;
        } else if (hour >= 10 && hour <= 16) {
            return 5;
        } else if (hour >= 20 && hour <= 22) {
            return 3;
        } else {
            return 1;
        }
    }

    updateVideoFrame(frameBase64) {
        const videoImg = document.getElementById('dashboard-video');
        const placeholder = document.querySelector('.video-placeholder');
        
        if (videoImg && frameBase64) {
            videoImg.src = `data:image/jpeg;base64,${frameBase64}`;
            videoImg.style.display = 'block';
            if (placeholder) placeholder.style.display = 'none';
        }
    }

    updateRealtimeInfo(data) {
        // 현재 차량 수
        const currentVehicles = document.getElementById('current-vehicles');
        if (currentVehicles) {
            currentVehicles.textContent = data.vehicle_count || 0;
        }

        // 평균 속도
        const avgSpeed = document.getElementById('avg-speed');
        if (avgSpeed) {
            avgSpeed.textContent = (data.avg_speed || 0).toFixed(1);
        }

        // 교통 상황
        const trafficLevel = document.getElementById('traffic-level');
        if (trafficLevel && data.traffic_level) {
            trafficLevel.textContent = data.traffic_level;
            trafficLevel.className = 'traffic-level-badge';
            
            if (data.traffic_level === '원활') trafficLevel.classList.add('smooth');
            else if (data.traffic_level === '보통') trafficLevel.classList.add('normal');
            else if (data.traffic_level === '혼잡') trafficLevel.classList.add('congested');
        }
    }

    updateTrafficData(data) {
        console.log('updateTrafficData 호출:', data); // 디버깅용
        
        if (data.minute_entered !== undefined && data.minute_exited !== undefined) {
            this.trafficData.totalIn += data.minute_entered;
            this.trafficData.totalOut += data.minute_exited;
            
            // 차트 업데이트
            const timeLabel = new Date().toLocaleTimeString('ko-KR', { 
                hour: '2-digit', 
                minute: '2-digit' 
            });
            
            if (this.chart) {
                console.log('차트 업데이트:', data.minute_entered, data.minute_exited); // 디버깅용
                this.chart.data.labels.push(timeLabel);
                this.chart.data.datasets[0].data.push(data.minute_entered);
                this.chart.data.datasets[1].data.push(data.minute_exited);
                
                // 최대 20개 데이터만 유지
                if (this.chart.data.labels.length > 20) {
                    this.chart.data.labels.shift();
                    this.chart.data.datasets[0].data.shift();
                    this.chart.data.datasets[1].data.shift();
                }
                
                this.chart.update('none'); // 애니메이션 없이 업데이트
            } else {
                console.log('차트가 없어서 업데이트 실패'); // 디버깅용
            }
            
            // 새로운 시간대 데이터 추가
            this.trafficData.hourlyData.push({
                time: timeLabel,
                in: data.minute_entered,
                out: data.minute_exited
            });
            
            // 최대 12개 항목만 유지
            if (this.trafficData.hourlyData.length > 12) {
                this.trafficData.hourlyData.shift();
            }
            
            this.updateFlowTable();
            this.updateFlowSummary();
        }

        // 총 누적 카운트 업데이트
        if (data.total_entered !== undefined) {
            this.trafficData.totalIn = data.total_entered;
        }
        if (data.total_exited !== undefined) {
            this.trafficData.totalOut = data.total_exited;
        }
        
        this.updateFlowSummary();
    }

    updateFlowTable() {
        const tableBody = document.getElementById('flow-table-body');
        if (!tableBody) return;

        const now = new Date();
        const currentTime = now.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });

        tableBody.innerHTML = this.trafficData.hourlyData
            .slice(-8) // 최근 8개만 표시
            .reverse()
            .map(data => {
                const isCurrentTime = data.time === currentTime;
                return `
                    <tr class="${isCurrentTime ? 'current-hour' : ''}">
                        <td>${data.time}</td>
                        <td>${data.in}</td>
                        <td>${data.out}</td>
                    </tr>
                `;
            }).join('');
    }

    updateFlowSummary() {
        const totalInElement = document.getElementById('total-in');
        const totalOutElement = document.getElementById('total-out');
        
        if (totalInElement) totalInElement.textContent = this.trafficData.totalIn;
        if (totalOutElement) totalOutElement.textContent = this.trafficData.totalOut;
    }

    startMonitoring() {
        const startBtn = document.getElementById('start-monitoring');
        const stopBtn = document.getElementById('stop-monitoring');
        const statusBadge = document.getElementById('status-badge');
        
        fetch(`/api/cctv/${this.cctvId}/start`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('모니터링 시작:', data);
                this.isMonitoring = true;
                
                if (startBtn) startBtn.style.display = 'none';
                if (stopBtn) stopBtn.style.display = 'flex';
                if (statusBadge) {
                    statusBadge.textContent = '활성';
                    statusBadge.className = 'status-badge active';
                }
                
                this.updateSystemStatus('camera', 'active');
                this.updateSystemStatus('ai', 'active');
            })
            .catch(error => {
                console.error('모니터링 시작 실패:', error);
                alert('모니터링 시작에 실패했습니다.');
            });
    }

    stopMonitoring() {
        const startBtn = document.getElementById('start-monitoring');
        const stopBtn = document.getElementById('stop-monitoring');
        const statusBadge = document.getElementById('status-badge');
        
        fetch(`/api/cctv/${this.cctvId}/stop`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                console.log('모니터링 중지:', data);
                this.isMonitoring = false;
                
                if (startBtn) startBtn.style.display = 'flex';
                if (stopBtn) stopBtn.style.display = 'none';
                if (statusBadge) {
                    statusBadge.textContent = '비활성';
                    statusBadge.className = 'status-badge inactive';
                }
                
                this.updateSystemStatus('camera', 'inactive');
                this.updateSystemStatus('ai', 'inactive');
            })
            .catch(error => {
                console.error('모니터링 중지 실패:', error);
                alert('모니터링 중지에 실패했습니다.');
            });
    }

    updateSystemStatus(component, status) {
        const statusMap = {
            'camera': { indicator: 'camera-indicator', value: 'camera-status' },
            'ai': { indicator: 'ai-indicator', value: 'ai-status' },
            'network': { indicator: 'network-indicator', value: 'network-status' }
        };

        const elements = statusMap[component];
        if (!elements) return;

        const indicator = document.getElementById(elements.indicator);
        const valueElement = document.getElementById(elements.value);

        if (indicator) {
            indicator.className = `status-indicator ${status}`;
        }

        if (valueElement) {
            const statusText = {
                'active': '정상',
                'warning': '주의',
                'error': '오류',
                'inactive': '비활성'
            };
            valueElement.textContent = statusText[status] || status;
        }
    }

    resetData() {
        if (confirm('모든 데이터를 초기화하시겠습니까?')) {
            this.trafficData = { totalIn: 0, totalOut: 0, hourlyData: [] };
            
            // 차트 데이터도 리셋
            if (this.chart) {
                this.chart.data.labels = [];
                this.chart.data.datasets[0].data = [];
                this.chart.data.datasets[1].data = [];
                this.chart.update();
            }
            
            this.generateInitialData();
            this.updateFlowTable();
            this.updateFlowSummary();
            console.log('데이터 리셋 완료');
        }
    }

    exportData() {
        const data = {
            cctv_id: this.cctvId,
            cctv_info: this.cctvInfo,
            traffic_data: this.trafficData,
            export_time: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cctv_${this.cctvId}_data_${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);
        
        console.log('데이터 내보내기 완료');
    }
}

// 페이지 로드 시 초기화
document.addEventListener('DOMContentLoaded', () => {
    const cctvId = parseInt(window.location.pathname.split('/').pop());
    const cctvInfo = window.cctvInfo || { name: `CCTV ${cctvId}` };
    
    console.log('대시보드 초기화:', cctvId, cctvInfo); // 디버깅용
    window.dashboard = new SimpleDashboard(cctvId, cctvInfo);
});