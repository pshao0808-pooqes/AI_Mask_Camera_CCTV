# 🚦 AI 기반 CCTV 통합 플랫폼

> **실시간 객체 탐지 및 영상 분석을 위한 종합 CCTV 모니터링 시스템**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Flask-2.0+-red.svg)](https://flask.palletsprojects.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-orange.svg)](https://streamlit.io)

## 📋 프로젝트 개요

AI 기반 CCTV 통합 플랫폼은 실시간 차량 탐지, 교통량 분석, 속도 측정을 통한 지능형 교통 모니터링 시스템입니다. 서울시 CCTV 데이터를 활용하여 실시간 교통 상황을 분석하고 웹 기반 대시보드로 시각화합니다.

### ✨ 주요 기능

- 🚗 **실시간 차량 탐지**: YOLOv8을 활용한 승용차, 버스, 트럭, 오토바이 구분
- 📊 **교통량 분석**: 진입/진출 차량 카운팅 및 시간별 통계
- ⚡ **속도 측정**: 차량 궤적 추적을 통한 실시간 속도 계산
- 🗺️ **지도 기반 모니터링**: Leaflet.js를 활용한 실시간 CCTV 위치 표시
- 📈 **실시간 대시보드**: Chart.js를 활용한 동적 데이터 시각화
- 💻 **다중 모드 지원**: CCTV, 카메라, 비디오 분석 모드

## 🏗️ 시스템 구조

```
📁 AI-CCTV-Platform/
├── 📁 backend/
│   ├── app.py                    # Flask 메인 서버
│   ├── cctv_analyzer.py         # CCTV 분석 엔진
│   ├── database.py              # 데이터베이스 관리
│   └── traffic_api.py           # 교통 API 연동
├── 📁 frontend/
│   ├── index.html               # 메인 대시보드
│   ├── dashboard.html           # 개별 CCTV 대시보드
│   └── static/
│       ├── css/                 # 스타일시트
│       └── js/                  # JavaScript 파일
├── 📁 video_analysis/
│   ├── video.py                 # 비디오 분석 엔진
│   ├── camera.py                # 카메라 제어
│   └── ui_video.py              # 비디오 분석 UI
├── 📁 video_loader/
│   ├── video_loader.py          # 비디오 로더
│   └── ui_video_loader.py       # 비디오 UI
└── main.py                      # 통합 실행 파일
```

## 🚀 빠른 시작

### 1. 환경 요구사항

- **Python**: 3.8 이상
- **운영체제**: Windows 10/11, macOS, Linux
- **메모리**: 최소 8GB RAM (GPU 사용 시 4GB VRAM 권장)
- **디스크**: 최소 2GB 여유 공간

### 2. 설치 방법

```bash
# 저장소 클론
git clone https://github.com/pshao0808-pooqes/AI_Mask_Camera_CCTV.git
cd ai-cctv-platform

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 3. 의존성 패키지

```txt
# requirements.txt
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
flask>=2.3.0
flask-cors>=4.0.0
flask-socketio>=5.3.0
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
requests>=2.31.0
plotly>=5.15.0
leaflet>=0.8.0
```

### 4. 실행 방법

#### Option 1: 통합 실행 (권장)
```bash
streamlit run main.py
```

#### Option 2: 개별 모드 실행
```bash
# CCTV 모니터링 모드
cd backend
python app.py

# 비디오 분석 모드
cd video_analysis
streamlit run ui_video.py

# 영상 로더 모드
cd video_loader
streamlit run ui_video_loader.py
```

## 📱 사용 방법

### 1. CCTV 모니터링

1. **서버 시작**: 통합 실행기에서 "CCTV 모드" 선택
2. **브라우저 열기**: `http://localhost:5000` 접속
3. **CCTV 선택**: 좌측 목록에서 모니터링할 CCTV 선택
4. **실시간 분석**: 차량 탐지 및 교통량 분석 시작

### 2. 비디오 분석

1. **비디오 업로드**: 지원 형식(MP4, MOV, AVI, MKV, WEBM)
2. **분석 설정**: 신뢰도 임계값, 모델 선택
3. **실시간 분석**: 사람 탐지, 세그멘테이션, 마스크 탐지
4. **결과 확인**: 프레임별 분석 결과 및 통계

### 3. 영상 모니터

1. **파일 업로드**: 드래그 앤 드롭 또는 파일 선택
2. **품질 분석**: 해상도, 프레임율, 비트레이트 분석
3. **최적화 제안**: 화질 개선 및 압축 권장사항

## ⚙️ 설정 및 최적화

### CPU 최적화 설정

```python
# CPU 환경에서 성능 최적화
def _detect_device(self):
    if device_info['device'] == 'cpu':
        cpu_count = torch.get_num_threads()
        optimal_threads = min(8, max(4, cpu_count // 2))
        torch.set_num_threads(optimal_threads)
        
        # 프레임 리사이즈로 처리 속도 향상
        frame_resized = cv2.resize(frame, (640, 480))
```

### GPU 설정

```python
# GPU 사용 시 자동 최적화
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model.to('cuda')
```

### API 키 설정

```python
# 환경변수 설정 (권장)
import os
os.environ['ITS_API_KEY'] = 'your_api_key_here'
os.environ['KOROAD_API_KEY'] = 'your_api_key_here'
```

## 🔧 API 연동

### 서울시 CCTV API

```python
# ITS 한국도로공사 API 사용
url = "https://openapi.its.go.kr:9443/cctvInfo"
params = {
    "apiKey": "your_api_key",
    "type": "ex",
    "cctvType": "4",
    "minX": "126.76",  # 서울 서쪽 경계
    "maxX": "127.18",  # 서울 동쪽 경계
    "minY": "37.41",   # 서울 남쪽 경계
    "maxY": "37.70"    # 서울 북쪽 경계
}
```

### 교통사고 API

```python
# 도로교통공단 API (선택사항)
base_url = "https://opendata.koroad.or.kr/api/rest"
```

## 📊 성능 지표

### 처리 성능

| 환경 | 해상도 | 평균 처리시간 | 실시간 FPS |
|------|--------|---------------|------------|
| GPU (RTX 3080) | 1920x1080 | ~0.034초 | ~30fps |
| CPU (Intel i7) | 640x480 | ~0.15초 | ~10fps |

### 시스템 요구사항

| 구성요소 | 최소사양 | 권장사양 |
|----------|----------|----------|
| CPU | Intel i5-8400 | Intel i7-10700K |
| RAM | 8GB | 16GB |
| GPU | - | RTX 3060 이상 |
| 저장공간 | 2GB | 10GB |

## 🔍 주요 알고리즘

### 1. 차량 탐지
- **모델**: YOLOv8 (n/s/m/l/x 버전 지원)
- **클래스**: 승용차(2), 오토바이(3), 버스(5), 트럭(7)
- **최적화**: CPU/GPU 환경별 동적 해상도 조정

### 2. 차량 추적
- **알고리즘**: Centroid Tracker
- **매칭**: 유클리드 거리 기반 Greedy 매칭
- **생명주기**: 30프레임 미감지 시 트랙 삭제

### 3. 속도 계산
```python
# 속도 계산 공식
distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)  # 픽셀 거리
speed = (distance * scale_factor * fps * 3.6) / 1000  # km/h
```

## 🌐 웹 인터페이스

### 메인 대시보드
- **지도**: Leaflet.js 기반 실시간 CCTV 위치
- **영상**: 멀티 CCTV 동시 모니터링
- **통계**: 실시간 교통량 및 속도 표시

### 개별 CCTV 대시보드
- **차트**: Chart.js 기반 실시간 데이터 시각화
- **제어**: 감지 민감도, 데이터 리셋 기능
- **내보내기**: CSV 형태 데이터 다운로드

## 🛠️ 개발자 가이드

### 프로젝트 구조 설명

```python
# 주요 클래스 구조
class CCTVAnalyzer:
    - detect_vehicles()      # 차량 탐지
    - track_vehicles()       # 차량 추적
    - calculate_speed()      # 속도 계산
    - get_statistics()       # 통계 데이터

class VehicleTracker:
    - update()              # 트랙 업데이트
    - calculate_distance()   # 거리 계산
    - count_vehicles()      # 차량 카운팅
```

### 커스터마이징

1. **새로운 클래스 추가**
```python
# cctv_analyzer.py에서 클래스 ID 추가
classes = [2, 3, 5, 7, 새로운_클래스_ID]
class_names = {새로운_클래스_ID: '새로운_차량_유형'}
```

2. **새로운 지역 추가**
```python
# 좌표 범위 수정
params = {
    "minX": "새로운_서쪽_경계",
    "maxX": "새로운_동쪽_경계",
    "minY": "새로운_남쪽_경계", 
    "maxY": "새로운_북쪽_경계"
}
```

## 🐛 문제 해결

### 자주 발생하는 문제

1. **CUDA 메모리 오류**
```bash
# 해결방법: GPU 메모리 정리
torch.cuda.empty_cache()
```

2. **모델 로딩 실패**
```bash
# 해결방법: 모델 재다운로드
pip uninstall ultralytics
pip install ultralytics
```

3. **API 호출 실패**
```bash
# 해결방법: API 키 및 네트워크 확인
curl -X GET "https://openapi.its.go.kr:9443/cctvInfo?apiKey=YOUR_KEY"
```

### 로그 확인

```python
# 디버그 모드 활성화
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 향후 개발 계획

### v2.0 로드맵
- [ ] **마이크로서비스 아키텍처** 전환
- [ ] **엣지 컴퓨팅** 지원
- [ ] **사람 탐지** 기능 추가
- [ ] **이상행동 탐지** 구현
- [ ] **번호판 인식** 기능
- [ ] **날씨 감지** 연동

### v2.1 계획
- [ ] **모바일 앱** 개발
- [ ] **AI 모델 개선** (커스텀 데이터셋)
- [ ] **다중 도시** 지원 확장
- [ ] **클라우드 배포** 지원

## 👥 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **Ultralytics**: YOLOv8 모델 제공
- **한국도로공사**: CCTV 데이터 API 제공
- **도로교통공단**: 교통사고 데이터 API 제공
- **OpenCV Community**: 영상 처리 라이브러리



---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
