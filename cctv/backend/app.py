from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room  # join_room 추가
import threading
import time
from cctv_analyzer import CCTVAnalyzer
from database import TrafficDatabase
import webbrowser
import sys
import os
from pathlib import Path
from traffic_api import TrafficAccidentAPI



app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend')
app.config['SECRET_KEY'] = 'your-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# 전역 객체
analyzer = CCTVAnalyzer()
db = TrafficDatabase()
monitoring_threads = {}
accident_api = TrafficAccidentAPI()


@app.route('/api/accidents/data')
def get_accident_data():
    # 서울 전체 데이터 가져오기
    accidents = accident_api.get_accident_data(sido_code="11", gugun_code="")
    return jsonify({'success': True, 'data': accidents, 'count': len(accidents)})


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard/<int:cctv_id>')
def dashboard(cctv_id):
    """CCTV 대시보드 페이지"""
    cctv_list = analyzer.get_seoul_cctv()
    cctv_info = next((cctv for cctv in cctv_list if cctv['id'] == cctv_id), None)
    
    if not cctv_info:
        return "CCTV를 찾을 수 없습니다.", 404
    
    return render_template('dashboard.html', cctv=cctv_info)

@app.route('/api/cctv/list')
def get_cctv_list():
    """CCTV 목록 반환"""
    cctv_list = analyzer.get_seoul_cctv()
    return jsonify(cctv_list)

@app.route('/api/cctv/<int:cctv_id>/start', methods=['POST'])
def start_cctv_monitoring(cctv_id):
    """특정 CCTV 모니터링 시작"""
    cctv_list = analyzer.get_seoul_cctv()
    cctv_info = next((cctv for cctv in cctv_list if cctv['id'] == cctv_id), None)
    
    if not cctv_info:
        return jsonify({'error': 'CCTV not found'}), 404
    
    if cctv_id in monitoring_threads:
        return jsonify({'message': 'Already monitoring'}), 200
    
    analyzer.is_running = True
    thread = threading.Thread(
        target=analyzer.start_monitoring, 
        args=(cctv_info, socketio),
        daemon=True
    )
    thread.start()
    monitoring_threads[cctv_id] = thread
    
    return jsonify({'message': 'Monitoring started'})

@app.route('/api/cctv/<int:cctv_id>/stop', methods=['POST'])
def stop_cctv_monitoring(cctv_id):
    """특정 CCTV 모니터링 중지"""
    analyzer.stop_monitoring(cctv_id)
    if cctv_id in monitoring_threads:
        del monitoring_threads[cctv_id]
    
    return jsonify({'message': 'Monitoring stopped'})

@app.route('/api/traffic/current')
def get_current_traffic():
    """현재 트래픽 상태 반환"""
    return jsonify(dict(analyzer.traffic_data))

@app.route('/api/traffic/history/<int:cctv_id>')
def get_traffic_history(cctv_id):
    """트래픽 히스토리 반환"""
    hours = request.args.get('hours', 24, type=int)
    history = db.get_traffic_history(cctv_id, hours)
    return jsonify(history)

@app.route('/api/cctv/<int:cctv_id>/stats')
def get_cctv_stats(cctv_id):
    """CCTV 통계 데이터 반환"""
    stats = analyzer.get_statistics(cctv_id)
    current_data = analyzer.traffic_data.get(cctv_id, {})
    
    # 시간별 데이터 생성 (실제로는 DB에서 가져와야 함)
    hourly_data = []
    for hour in range(24):
        hourly_data.append({
            'hour': hour,
            'vehicle_count': max(0, stats.get('entering', 0) + (hour * 2) - 20),
            'avg_speed': 25 + (hour % 5) * 5,
            'traffic_level': '원활' if hour < 7 or hour > 22 else ('보통' if hour < 17 else '혼잡')
        })
    
    # 차량 유형별 데이터
    vehicle_types = {
        '승용차': stats.get('entering', 0) * 0.7,
        '버스': stats.get('entering', 0) * 0.1,
        '트럭': stats.get('entering', 0) * 0.15,
        '오토바이': stats.get('entering', 0) * 0.05
    }
    
    return jsonify({
        'current': current_data,
        'stats': stats,
        'hourly_data': hourly_data,
        'vehicle_types': vehicle_types,
        'peak_hours': [8, 9, 17, 18, 19],
        'avg_daily_traffic': stats.get('entering', 0) + stats.get('exiting', 0)
    })

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to server'})
    # 디바이스 정보 전송
    emit('device_info', analyzer.get_device_info())

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('request_cctv_list')
def handle_cctv_list_request():
    """클라이언트에서 CCTV 목록 요청"""
    cctv_list = analyzer.get_seoul_cctv()
    emit('cctv_list', cctv_list)

@socketio.on('join_dashboard')
def handle_join_dashboard(data):
    """대시보드 룸 참가"""
    cctv_id = data['cctv_id']
    room = f'dashboard_{cctv_id}'
    join_room(room)
    emit('joined_dashboard', {'cctv_id': cctv_id})

def open_browser():
    """브라우저 자동 실행"""
    time.sleep(1.5)  # 서버 시작 대기
    webbrowser.open('http://localhost:5000')

def select_model():
    """사용할 모델 선택"""
    print("\n사용 가능한 모델:")
    models = [
        ("1", "YOLOv8n (가장 빠름, 가장 낮은 정확도)"),
        ("2", "YOLOv8s (빠름, 낮은 정확도)"),
        ("3", "YOLOv8m (보통 속도, 보통 정확도)"),
        ("4", "YOLOv8l (느림, 높은 정확도)"),
        ("5", "YOLOv8x (가장 느림, 가장 높은 정확도)")
    ]
    
    print("\n모델 성능 비교:")
    print("속도: n > s > m > l > x")
    print("정확도: x > l > m > s > n")
    print("-" * 50)
    
    for id, desc in models:
        print(f"{id}. {desc}")
        
    while True:
        choice = input("\n모델을 선택하세요 (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            return {
                "1": "YOLOv8n",
                "2": "YOLOv8s",
                "3": "YOLOv8m",
                "4": "YOLOv8l",
                "5": "YOLOv8x"
            }[choice]
        print("잘못된 선택입니다. 1-5 사이의 숫자를 입력하세요.")

if __name__ == '__main__':
    port = 5000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"잘못된 포트 번호: {sys.argv[1]}")
            port = 5000
    
    print(f"서버를 포트 {port}에서 시작합니다...")
    try:
        socketio.run(app, debug=False, host='0.0.0.0', port=port)
    except KeyboardInterrupt:
        print("\n시스템 종료 중...")
        analyzer.stop_all_monitoring()
        print("시스템이 종료되었습니다.")
