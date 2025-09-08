import cv2
import numpy as np
from ultralytics import YOLO
import requests
import xml.etree.ElementTree as ET
import threading
import time
from collections import defaultdict, deque
import base64
from datetime import datetime
import torch
import subprocess
import platform

class VehicleTracker:
    """개선된 차량 추적 클래스"""
    def __init__(self):
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = 10
        self.max_distance = 80
        
        # 카운팅을 위한 상태
        self.counted_vehicles = set()  # 이미 카운트된 차량 ID들
        
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def get_centroid(self, bbox):
        x1, y1, x2, y2 = bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def update_tracks(self, detections):
        if not detections:
            # 기존 트랙들의 disappeared 증가
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['disappeared'] += 1
                if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                    self.counted_vehicles.discard(track_id)  # 카운트 기록에서 제거
                    del self.tracks[track_id]
            return []
        
        # 새로운 detection들의 중심점 계산
        input_centroids = []
        for detection in detections:
            centroid = self.get_centroid(detection['bbox'])
            input_centroids.append({
                'centroid': centroid,
                'detection': detection
            })
        
        # 기존 트랙이 없으면 새로 생성
        if len(self.tracks) == 0:
            for input_data in input_centroids:
                self.tracks[self.next_id] = {
                    'centroid': input_data['centroid'],
                    'detection': input_data['detection'],
                    'disappeared': 0,
                    'trajectory': deque([input_data['centroid']], maxlen=20)
                }
                self.next_id += 1
        else:
            # 기존 트랙과 새로운 detection 매칭
            track_centroids = []
            track_ids = []
            
            for track_id, track_data in self.tracks.items():
                if track_data['disappeared'] < self.max_disappeared:
                    track_centroids.append(track_data['centroid'])
                    track_ids.append(track_id)
            
            if len(track_centroids) > 0:
                # 거리 행렬 계산
                distance_matrix = np.zeros((len(track_centroids), len(input_centroids)))
                for i, track_centroid in enumerate(track_centroids):
                    for j, input_data in enumerate(input_centroids):
                        distance_matrix[i, j] = self.calculate_distance(
                            track_centroid, input_data['centroid']
                        )
                
                # 헝가리안 알고리즘 대신 간단한 greedy 매칭
                used_row_indices = set()
                used_col_indices = set()
                
                # 거리순으로 정렬하여 매칭
                matches = []
                for i in range(len(track_centroids)):
                    for j in range(len(input_centroids)):
                        matches.append((distance_matrix[i, j], i, j))
                
                matches.sort()
                
                for distance, i, j in matches:
                    if (i not in used_row_indices and 
                        j not in used_col_indices and 
                        distance <= self.max_distance):
                        
                        track_id = track_ids[i]
                        new_centroid = input_centroids[j]['centroid']
                        new_detection = input_centroids[j]['detection']
                        
                        # 트랙 업데이트
                        self.tracks[track_id]['centroid'] = new_centroid
                        self.tracks[track_id]['detection'] = new_detection
                        self.tracks[track_id]['disappeared'] = 0
                        self.tracks[track_id]['trajectory'].append(new_centroid)
                        
                        used_row_indices.add(i)
                        used_col_indices.add(j)
                
                # 매칭되지 않은 기존 트랙들의 disappeared 증가
                for i, track_id in enumerate(track_ids):
                    if i not in used_row_indices:
                        self.tracks[track_id]['disappeared'] += 1
                
                # 매칭되지 않은 새로운 detection들을 새 트랙으로 생성
                for j in range(len(input_centroids)):
                    if j not in used_col_indices:
                        self.tracks[self.next_id] = {
                            'centroid': input_centroids[j]['centroid'],
                            'detection': input_centroids[j]['detection'],
                            'disappeared': 0,
                            'trajectory': deque([input_centroids[j]['centroid']], maxlen=20)
                        }
                        self.next_id += 1
            else:
                # 모든 기존 트랙이 사라진 경우 새로 생성
                for input_data in input_centroids:
                    self.tracks[self.next_id] = {
                        'centroid': input_data['centroid'],
                        'detection': input_data['detection'],
                        'disappeared': 0,
                        'trajectory': deque([input_data['centroid']], maxlen=20)
                    }
                    self.next_id += 1
        
        # 너무 오래 사라진 트랙들 제거
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['disappeared'] > self.max_disappeared:
                self.counted_vehicles.discard(track_id)
                del self.tracks[track_id]
        
        # 활성 트랙들 반환
        active_tracks = []
        for track_id, track_data in self.tracks.items():
            if track_data['disappeared'] == 0:
                detection = track_data['detection'].copy()
                detection['track_id'] = track_id
                detection['trajectory'] = list(track_data['trajectory'])
                active_tracks.append(detection)
        
        return active_tracks

class CCTVAnalyzer:
    def __init__(self):
        self.model = "YOLOv8s"  # 기본 모델
        # GPU/CPU 감지 및 설정
        self.device_info = self._detect_device()
        self.device = self.device_info['device']
        
        # YOLO 모델 로드
        self.set_model(self.model)  # 수정된 set_model 메서드 사용
    
        self.cctv_streams = {}
        self.traffic_data = defaultdict(dict)
        self.vehicle_trackers = {}
        self.is_running = False
        
        # 차량 카운팅 데이터
        self.vehicle_counts = defaultdict(lambda: {'entering': 0, 'exiting': 0, 'current': 0})
        self.minute_counts = defaultdict(lambda: {'entering': 0, 'exiting': 0})
        self.last_minute_reset = defaultdict(int)
    def _detect_device(self):
        """GPU/CPU 감지 및 최적 디바이스 선택"""
        device_info = {
            'device': 'cpu',
            'type': 'CPU',
            'name': 'CPU 최적화 모드',
            'details': {}
        }
        
        print("=== 디바이스 감지 시작 ===")
        
        # CUDA 사용 가능성 체크
        cuda_available = torch.cuda.is_available()
        print(f"PyTorch CUDA 지원: {cuda_available}")
        
        if cuda_available:
            try:
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    
                    # GPU 메모리 테스트
                    test_tensor = torch.randn(100, 100).cuda()
                    del test_tensor
                    torch.cuda.empty_cache()
                    
                    device_info = {
                        'device': 'cuda',
                        'type': 'GPU',
                        'name': gpu_name,
                        'details': {
                            'memory_total': f"{gpu_memory:.1f} GB",
                            'cuda_version': torch.version.cuda,
                            'device_count': device_count
                        }
                    }
                    
                    torch.backends.cudnn.benchmark = True
                    print(f"GPU 사용: {gpu_name}")
                    
            except Exception as e:
                print(f"GPU 초기화 실패, CPU 사용: {e}")
        
        if device_info['device'] == 'cpu':
            cpu_count = torch.get_num_threads()
            optimal_threads = min(8, max(4, cpu_count // 2))
            torch.set_num_threads(optimal_threads)
            device_info['details']['threads'] = optimal_threads
            print(f"CPU 최적화: {optimal_threads} 스레드 사용")
        
        print(f"최종 선택된 디바이스: {device_info['device']}")
        print("=== 디바이스 감지 완료 ===\n")
        
        return device_info
        
    def get_device_info(self):
        """현재 사용 중인 디바이스 정보 반환"""
        return self.device_info.copy()

    def get_seoul_cctv(self):
        """서울 CCTV 정보 가져오기"""
        url = "https://openapi.its.go.kr:9443/cctvInfo"
        params = {
            "apiKey": "690d3426b85f48779a508ad142198424",
            "type": "ex",
            "cctvType": "4",
            "minX": "126.76",
            "maxX": "127.18", 
            "minY": "37.41",
            "maxY": "37.70",
            "getType": "xml"
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return []

            root = ET.fromstring(response.content)
            cctv_list = []

            for item in root.findall(".//data"):
                cctv_name = item.findtext("cctvname")
                cctv_url = item.findtext("cctvurl")
                coord_x = item.findtext("coordx")
                coord_y = item.findtext("coordy")
                
                if cctv_url and coord_x and coord_y:
                    try:
                        lat = float(coord_y)
                        lon = float(coord_x)
                        cctv_list.append({
                            'id': len(cctv_list),
                            'name': cctv_name or f"CCTV_{len(cctv_list)}",
                            'url': cctv_url,
                            'lat': lat,
                            'lon': lon,
                            'status': 'inactive'
                        })
                    except ValueError:
                        continue

            return cctv_list[:20]

        except Exception as e:
            print(f"CCTV 정보 가져오기 실패: {e}")
            return []

    def detect_vehicles(self, frame):
        """프레임에서 차량 감지"""
        try:
            if self.device == 'cpu':
                frame_resized = cv2.resize(frame, (640, 480))
                results = self.yolo_model(frame_resized, classes=[2, 3, 5, 7], verbose=False, conf=0.3)
                scale_x = frame.shape[1] / 640
                scale_y = frame.shape[0] / 480
            else:
                results = self.yolo_model(frame, classes=[2, 3, 5, 7], verbose=False, conf=0.4)
                scale_x = scale_y = 1.0
            
            detections = []
            class_names = {2: '승용차', 3: '오토바이', 5: '버스', 7: '트럭'}
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        if self.device == 'cpu':
                            x1, x2 = x1 * scale_x, x2 * scale_x
                            y1, y2 = y1 * scale_y, y2 * scale_y
                        
                        
                        class_id = int(box.cls[0])
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            
                            'class_id': class_id,
                            'class_name': class_names.get(class_id, '차량')
                        })
            
            return detections
            
        except Exception as e:
            print(f"차량 감지 오류: {e}")
            return []

    def check_line_crossing(self, cctv_id, tracked_vehicles, frame_height):
        """상단선/하단선 교차 체크 및 카운팅"""
        if cctv_id not in self.vehicle_trackers:
            return {'entering': 0, 'exiting': 0, 'current_vehicles': 0}
        
        tracker = self.vehicle_trackers[cctv_id]
        counts = {'entering': 0, 'exiting': 0, 'current_vehicles': 0}
        
        # 분 단위 리셋 체크
        current_minute = int(time.time() / 60)
        if self.last_minute_reset[cctv_id] != current_minute:
            self.minute_counts[cctv_id] = {'entering': 0, 'exiting': 0}
            self.last_minute_reset[cctv_id] = current_minute
        
        # 라인 설정
        upper_line = int(frame_height * 0.3)
        lower_line = int(frame_height * 0.7)
        middle_area = int(frame_height * 0.5)  # 중간 영역 추가
        
        for vehicle in tracked_vehicles:
            track_id = vehicle.get('track_id')
            trajectory = vehicle.get('trajectory', [])
            
            if track_id is None or len(trajectory) < 2:
                continue
            
            current_y = trajectory[-1][1]
            prev_y = trajectory[-2][1]
            
            # 중간 영역에 있는 차량은 자동으로 추적 대상에 추가 (진출 카운팅을 위해)
            if upper_line < current_y < lower_line:
                tracker.counted_vehicles.add(track_id)
            
            # 상단선 교차 (진입)
            if prev_y <= upper_line < current_y:
                if track_id not in tracker.counted_vehicles:
                    counts['entering'] += 1
                    tracker.counted_vehicles.add(track_id)
                    self.vehicle_counts[cctv_id]['entering'] += 1
                    self.vehicle_counts[cctv_id]['current'] += 1
                    self.minute_counts[cctv_id]['entering'] += 1
            
            # 하단선 교차 (진출) - 조건 완화
            elif prev_y <= lower_line < current_y:
                # track_id in counted_vehicles 조건 제거하고 무조건 카운트
                counts['exiting'] += 1
                tracker.counted_vehicles.discard(track_id)  # 추적에서 제거
                self.vehicle_counts[cctv_id]['exiting'] += 1
                self.vehicle_counts[cctv_id]['current'] = max(0, self.vehicle_counts[cctv_id]['current'] - 1)
                self.minute_counts[cctv_id]['exiting'] += 1
                print(f"진출 카운트: 차량 ID {track_id}")  # 디버깅용
            
            # 역방향도 동일하게 처리
            elif current_y <= upper_line < prev_y:
                counts['exiting'] += 1
                self.vehicle_counts[cctv_id]['exiting'] += 1
                self.minute_counts[cctv_id]['exiting'] += 1
            elif current_y <= lower_line < prev_y:
                counts['entering'] += 1
                tracker.counted_vehicles.add(track_id)
                self.vehicle_counts[cctv_id]['entering'] += 1
                self.vehicle_counts[cctv_id]['current'] += 1
                self.minute_counts[cctv_id]['entering'] += 1
        
        counts['current_vehicles'] = len([v for v in tracked_vehicles if v.get('track_id') is not None])
        return counts

    def draw_korean_text(self, frame, text, position, font_scale=0.6, color=(255, 255, 255), thickness=2):
        """한글 텍스트를 이미지로 렌더링하여 프레임에 그리기"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            # PIL 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_image)
            
            # 기본 한글 폰트 시도
            try:
                font = ImageFont.truetype("malgun.ttf", int(20 * font_scale))
            except:
                try:
                    font = ImageFont.truetype("gulim.ttf", int(20 * font_scale))
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", int(20 * font_scale))
                    except:
                        font = ImageFont.load_default()
            
            # 텍스트 그리기
            x, y = position
            draw.text((x, y), text, font=font, fill=color)
            
            # 다시 OpenCV 형식으로 변환
            frame_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame_with_text
            
        except ImportError:
            # PIL이 없는 경우 기본 OpenCV 텍스트 사용
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return frame
        except Exception as e:
            # 오류 발생 시 기본 텍스트 사용
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            return frame

    def draw_detection_results(self, frame, tracked_vehicles, cctv_id):
        """test.py 스타일의 고급 검출 결과 그리기"""
        height, width = frame.shape[:2]
        
        # 상단선/하단선 설정
        upper_line = int(height * 0.3)
        lower_line = int(height * 0.7)
        
        # 상단선 (진입선) - 빨간색, 더 굵게
        cv2.line(frame, (0, upper_line), (width, upper_line), (0, 0, 255), 4)
        # 화살표 효과
        cv2.arrowedLine(frame, (20, upper_line), (60, upper_line), (0, 0, 255), 6)
        cv2.arrowedLine(frame, (width-60, upper_line), (width-20, upper_line), (0, 0, 255), 6)
        
        # 하단선 (퇴장선) - 파란색, 더 굵게  
        cv2.line(frame, (0, lower_line), (width, lower_line), (255, 0, 0), 4)
        cv2.arrowedLine(frame, (20, lower_line), (60, lower_line), (255, 0, 0), 6)
        cv2.arrowedLine(frame, (width-60, lower_line), (width-20, lower_line), (255, 0, 0), 6)
        
        # 라인 라벨 (한글)
        frame = self.draw_korean_text(frame, "차량 계수선", (width//2 - 60, upper_line - 25), 
                                    font_scale=0.8, color=(0, 0, 255), thickness=2)
        
        # 차량 타입별 색상 (test.py와 동일)
        vehicle_colors = {
            '승용차': (0, 255, 0),    # 초록색
            '오토바이': (0, 165, 255), # 주황색  
            '버스': (255, 255, 0),    # 청록색
            '트럭': (255, 0, 255)     # 마젠타색
        }
        
        # 차량별 상세 검출 박스 그리기 (test.py 스타일)
        for vehicle in tracked_vehicles:
            bbox = vehicle['bbox']
            x1, y1, x2, y2 = bbox
            
            class_name = vehicle['class_name']
            track_id = vehicle.get('track_id', '?')
            trajectory = vehicle.get('trajectory', [])
            
            color = vehicle_colors.get(class_name, (128, 128, 128))
            
            # 이동 궤적 그리기 (선으로 연결)
            if len(trajectory) > 1:
                trajectory_color = tuple(int(c * 0.7) for c in color)
                for i in range(1, len(trajectory)):
                    pt1 = trajectory[i-1]
                    pt2 = trajectory[i]
                    cv2.line(frame, pt1, pt2, trajectory_color, 2)
                
                # 궤적 점들 표시
                for i, point in enumerate(trajectory):
                    alpha = 0.3 + (i / len(trajectory)) * 0.7
                    point_color = tuple(int(c * alpha) for c in color)
                    cv2.circle(frame, point, 3, point_color, -1)
            
            # L자형 모서리 박스 그리기 (더 멋진 스타일)
            self._draw_corner_box(frame, x1, y1, x2, y2, color, thickness=3, corner_length=30)
            
            # 중심점 표시 (더 크게)
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (center_x, center_y), 6, color, -1)
            cv2.circle(frame, (center_x, center_y), 8, (255, 255, 255), 2)
            
            # 간단한 라벨 (정확도 제거)
            label = f"{class_name} #{track_id}"
            
            # 배경 박스 크기 계산 (더 작게)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            bg_x1 = x1
            bg_y1 = y1 - text_size[1] - 10
            bg_x2 = x1 + text_size[0] + 8
            bg_y2 = y1 - 2
            
            # 화면 경계 체크
            if bg_y1 < 0:
                bg_y1 = y2 + 2
                bg_y2 = y2 + text_size[1] + 10
            
            # 반투명 배경 (더 투명하게)
            overlay = frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # 테두리 (더 얇게)
            cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), 1)
            
            # 한글 텍스트 그리기 (더 작게)
            text_y = bg_y1 + text_size[1] + 3 if bg_y1 > 0 else bg_y1 + text_size[1] + 8
            frame = self.draw_korean_text(frame, label, (x1 + 4, text_y - text_size[1]), 
                                        font_scale=0.5, color=(255, 255, 255), thickness=1)
        
       
        
        return frame
    
    def _draw_corner_box(self, frame, x1, y1, x2, y2, color, thickness=3, corner_length=25):
        """모서리 L자형 바운딩 박스 그리기"""
        # 좌상단 모서리
        cv2.line(frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
        
        # 우상단 모서리
        cv2.line(frame, (x2, y1), (x2 - corner_length, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
        
        # 좌하단 모서리
        cv2.line(frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_length), color, thickness)
        
        # 우하단 모서리
        cv2.line(frame, (x2, y2), (x2 - corner_length, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_length), color, thickness)
    
    

    def analyze_frame(self, frame, cctv_id):
        """프레임 분석 및 트래픽 정보 추출"""
        try:
            # CCTV별 차량 추적기 초기화
            if cctv_id not in self.vehicle_trackers:
                self.vehicle_trackers[cctv_id] = VehicleTracker()
            
            # 차량 감지
            detections = self.detect_vehicles(frame)
            
            # 차량 추적
            tracked_vehicles = self.vehicle_trackers[cctv_id].update_tracks(detections)
            
            # 라인 교차 체크
            crossing_data = self.check_line_crossing(cctv_id, tracked_vehicles, frame.shape[0])
            
            # 프레임에 결과 그리기
            annotated_frame = self.draw_detection_results(frame, tracked_vehicles, cctv_id)
            
            # 트래픽 레벨 계산
            current_count = len(tracked_vehicles)
            if current_count <= 5:
                traffic_level = "원활"
                color = "#4CAF50"
            elif current_count <= 15:
                traffic_level = "보통"  
                color = "#FFC107"
            else:
                traffic_level = "혼잡"
                color = "#F44336"
            
            # 평균 속도 계산 (간단한 추정)
            avg_speed = self.calculate_average_speed(tracked_vehicles)
            
            return {
                'vehicle_count': current_count,
                'traffic_level': traffic_level,
                'color': color,
                'vehicles': tracked_vehicles,
                'avg_speed': avg_speed,
                'annotated_frame': annotated_frame,
                'timestamp': datetime.now().isoformat(),
                'device': self.device.upper(),
                'crossing_data': crossing_data,
                'total_entered': self.vehicle_counts[cctv_id]['entering'],
                'total_exited': self.vehicle_counts[cctv_id]['exiting'],
                'current_in_area': self.vehicle_counts[cctv_id]['current']
            }

        except Exception as e:
            print(f"프레임 분석 오류: {e}")
            return None

    def calculate_average_speed(self, tracked_vehicles):
        """차량 평균 속도 계산 (픽셀 이동 기반 추정)"""
        if not tracked_vehicles:
            return 0.0
        
        total_speed = 0
        speed_count = 0
        
        for vehicle in tracked_vehicles:
            trajectory = vehicle.get('trajectory', [])
            if len(trajectory) >= 2:
                # 최근 두 점 사이의 거리 계산
                p1, p2 = trajectory[-2], trajectory[-1]
                pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                
                # 픽셀 거리를 속도로 변환 (간단한 추정)
                estimated_speed = min(80, max(10, pixel_distance * 0.8))
                total_speed += estimated_speed
                speed_count += 1
        
        return total_speed / speed_count if speed_count > 0 else 25.0

    def start_monitoring(self, cctv_info, socketio):
        """CCTV 모니터링 시작"""
        cctv_id = cctv_info['id']
        
        try:
            cap = cv2.VideoCapture(cctv_info['url'])
            if not cap.isOpened():
                print(f"CCTV {cctv_id} 연결 실패")
                return

            # 비디오 설정 최적화
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)

            self.cctv_streams[cctv_id] = {
                'cap': cap,
                'info': cctv_info,
                'active': True
            }

            frame_count = 0
            # 최적화 설정
            if self.device == 'cuda':
                frame_skip = 2
                sleep_time = 0.05
            else:
                frame_skip = 2
                sleep_time = 0.1
            
            print(f"CCTV {cctv_id} 모니터링 시작: {cctv_info['name']}")
            
            while self.is_running and self.cctv_streams[cctv_id]['active']:
                ret, frame = cap.read()
                if not ret:
                    print(f"CCTV {cctv_id} 프레임 읽기 실패")
                    break

                # 프레임 리사이즈
                if self.device == 'cpu':
                    frame = cv2.resize(frame, (800, 600))
                else:
                    frame = cv2.resize(frame, (1024, 768))

                if frame_count % frame_skip == 0:
                    analysis_result = self.analyze_frame(frame, cctv_id)
                    
                    if analysis_result:
                        self.traffic_data[cctv_id] = analysis_result
                        
                        # UI 업데이트용 데이터 전송
                        ui_data = {
                            'vehicle_count': analysis_result['vehicle_count'],
                            'traffic_level': analysis_result['traffic_level'],
                            'color': analysis_result['color'],
                            'avg_speed': analysis_result['avg_speed'],
                            'timestamp': analysis_result['timestamp'],
                            'device': analysis_result['device'],
                            'total_entered': analysis_result['total_entered'],
                            'total_exited': analysis_result['total_exited'],
                            'current_in_area': analysis_result['current_in_area'],
                            'minute_entered': self.minute_counts[cctv_id]['entering'],
                            'minute_exited': self.minute_counts[cctv_id]['exiting']
                        }
                        
                        socketio.emit('traffic_update', {
                            'cctv_id': cctv_id,
                            'data': ui_data
                        })

                        # 프레임 전송 (메인 페이지용)
                        _, buffer = cv2.imencode('.jpg', analysis_result['annotated_frame'], 
                                               [cv2.IMWRITE_JPEG_QUALITY, 85])
                        frame_base64 = base64.b64encode(buffer).decode('utf-8')
                        
                        socketio.emit('frame_update', {
                            'cctv_id': cctv_id,
                            'frame': frame_base64
                        })

                        # 대시보드용 프레임 전송
                        socketio.emit('dashboard_frame_update', {
                            'cctv_id': cctv_id,
                            'frame': frame_base64,
                            'data': ui_data
                        })

                        # 콘솔 출력 (디버깅용)
                        crossing = analysis_result['crossing_data']
                        if crossing['entering'] > 0 or crossing['exiting'] > 0:
                            print(f"CCTV {cctv_id}: 진입 +{crossing['entering']}, 퇴장 +{crossing['exiting']}, 현재 {analysis_result['current_in_area']}대")

                frame_count += 1
                time.sleep(sleep_time)

        except Exception as e:
            print(f"CCTV {cctv_id} 모니터링 오류: {e}")
        finally:
            if cctv_id in self.cctv_streams:
                self.cctv_streams[cctv_id]['cap'].release()
                del self.cctv_streams[cctv_id]
            print(f"CCTV {cctv_id} 모니터링 종료")

    def stop_monitoring(self, cctv_id):
        """특정 CCTV 모니터링 중지"""
        if cctv_id in self.cctv_streams:
            self.cctv_streams[cctv_id]['active'] = False
            
        # 차량 추적기 초기화
        if cctv_id in self.vehicle_trackers:
            del self.vehicle_trackers[cctv_id]
            
        # 카운트 데이터 유지 (리셋하지 않음)
        print(f"CCTV {cctv_id} 모니터링 중지")

    def stop_all_monitoring(self):
        """모든 CCTV 모니터링 중지"""
        self.is_running = False
        for cctv_id in list(self.cctv_streams.keys()):
            self.stop_monitoring(cctv_id)
        
        # 모든 추적기 정리
        self.vehicle_trackers.clear()
        print("모든 CCTV 모니터링 중지")

    def reset_counts(self, cctv_id=None):
        """카운트 데이터 리셋"""
        if cctv_id is None:
            # 모든 CCTV 카운트 리셋
            self.vehicle_counts.clear()
            print("모든 CCTV 카운트 리셋")
        else:
            # 특정 CCTV 카운트 리셋
            self.vehicle_counts[cctv_id] = {'entering': 0, 'exiting': 0, 'current': 0}
            if cctv_id in self.vehicle_trackers:
                self.vehicle_trackers[cctv_id].counted_vehicles.clear()
            print(f"CCTV {cctv_id} 카운트 리셋")

    def get_statistics(self, cctv_id=None):
        """통계 정보 반환"""
        if cctv_id is None:
            # 전체 통계
            total_stats = {'entering': 0, 'exiting': 0, 'current': 0}
            for stats in self.vehicle_counts.values():
                total_stats['entering'] += stats['entering']
                total_stats['exiting'] += stats['exiting']
                total_stats['current'] += stats['current']
            return total_stats
        else:
            # 특정 CCTV 통계
            return self.vehicle_counts.get(cctv_id, {'entering': 0, 'exiting': 0, 'current': 0})

    def set_model(self, model_name):
        """모델 설정"""
        self.model = model_name
        try:
            # 기존 모델이 있다면 메모리에서 해제
            if hasattr(self, 'yolo_model'):
                del self.yolo_model
            
            print(f"\n{model_name} 모델 초기화 중...")
            
            # YOLO 모델 이름 매핑
            model_mapping = {
                "YOLOv8n": "yolov8n.pt",
                "YOLOv8s": "yolov8s.pt",
                "YOLOv8m": "yolov8m.pt",
                "YOLOv8l": "yolov8l.pt",
                "YOLOv8x": "yolov8x.pt"
            }
            
            model_file = model_mapping.get(model_name)
            if not model_file:
                raise ValueError(f"지원되지 않는 모델: {model_name}")
                
            # 모델 로드 (없으면 자동 다운로드)
            print("모델 파일 확인 중...")
            self.yolo_model = YOLO(model_file)
            
            # GPU 사용 설정
            if self.device == 'cuda':
                print("GPU로 모델 이동 중...")
                self.yolo_model.to(self.device)
            
            print(f"{model_name} 모델 로드 완료!")
            return True
            
        except Exception as e:
            print(f"모델 로드 실패: {str(e)}")
            if "not found" in str(e).lower():
                print("모델 파일을 찾을 수 없습니다. 자동 다운로드를 시도합니다...")
                try:
                    from ultralytics import hub
                    model_file = f"ultralytics/{model_mapping.get(model_name)}"
                    self.yolo_model = YOLO(model_file)
                    if self.device == 'cuda':
                        self.yolo_model.to(self.device)
                    print(f"{model_name} 모델 다운로드 및 로드 완료!")
                    return True
                except Exception as download_error:
                    print(f"모델 다운로드 실패: {str(download_error)}")
                    return False
            return False
