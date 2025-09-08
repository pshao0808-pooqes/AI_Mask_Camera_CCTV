import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import threading
from queue import Queue
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
import gc
import pandas as pd

class VideoDetector:
    """개선된 하이브리드 접근법: YOLOv8-seg로 사람 탐지/세그먼테이션 + 얼굴 탐지 + mask_best.pt로 마스크 탐지"""
    
    def __init__(self):
        self.person_model = None      # YOLOv8 세그먼테이션 모델
        self.mask_model = None        # 마스크 탐지 전용 모델
        self.face_detector = None     # 얼굴 탐지 모델 추가
        self.use_hybrid_approach = True  # 하이브리드 접근법 활성화
        self.use_face_detection = True   # 얼굴 탐지 기능 활성화
        
        # 모델별 클래스 정보 저장
        self.mask_model_classes = {}
        
        self.performance_settings = {
            'detection_interval': 3,
            'confidence_threshold': 0.5,
            'target_fps': 30,
            'use_gpu_acceleration': torch.cuda.is_available(),
            'detection_resolution': 'medium'
        }
        
        # 탐지 결과 저장소
        self.detected_people = []
        self.segmented_people = []
        self.mask_wearers = []
        self.non_mask_people = []
        self.detected_faces = []  # 얼굴 탐지 결과 저장 추가
        self.last_detection_result = None
        
        # 각 단계별 색상 정의
        self.person_color = (0, 255, 0)      # 일반 사람용 초록색
        self.segment_color = (255, 165, 0)   # 세그먼테이션용 주황색
        self.mask_color = (0, 0, 255)        # 마스크 착용자용 빨간색
        self.no_mask_color = (255, 255, 0)   # 마스크 미착용자용 노란색
        self.face_color = (0, 255, 255)      # 얼굴 탐지용 시안색

    def debug_model_loading(self):
        """모델 로딩 문제 디버깅"""
        print("=== 모델 로딩 디버깅 시작 ===")
        
        # 현재 디렉토리 확인
        current_dir = os.getcwd()
        print(f"현재 작업 디렉토리: {current_dir}")
        
        # 파일 존재 여부 확인
        mask_model_path = 'mask_best.pt'
        mask_model_exists = os.path.exists(mask_model_path)
        print(f"mask_best.pt 파일 존재: {mask_model_exists}")
        
        if mask_model_exists:
            file_size = os.path.getsize(mask_model_path)
            print(f"파일 크기: {file_size} bytes")
            
            # 파일 읽기 권한 확인
            readable = os.access(mask_model_path, os.R_OK)
            print(f"파일 읽기 권한: {readable}")
        else:
            # 디렉토리 내 .pt 파일들 찾기
            pt_files = [f for f in os.listdir('.') if f.endswith('.pt')]
            print(f"발견된 .pt 파일들: {pt_files}")
        
        # PyTorch 및 YOLO 버전 확인
        print(f"PyTorch 버전: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name()}")
        
        try:
            from ultralytics import __version__ as yolo_version
            print(f"Ultralytics YOLO 버전: {yolo_version}")
        except:
            print("Ultralytics 버전 확인 실패")
        
        print("=== 디버깅 완료 ===\n")

    def debug_mask_model(self):
        """마스크 모델의 클래스 구조 디버깅"""
        if self.mask_model is None:
            print("마스크 모델이 로드되지 않았습니다.")
            return False
            
        print("=== 마스크 모델 정보 ===")
        print(f"모델 클래스 수: {len(self.mask_model.names)}")
        print(f"클래스 이름들: {self.mask_model.names}")
        
        # 클래스 정보를 저장
        self.mask_model_classes = self.mask_model.names
        
        # 예상되는 클래스 구조 확인
        for idx, name in self.mask_model_classes.items():
            print(f"  클래스 {idx}: {name}")
            
        print("=====================\n")
        return True

    def load_face_detector(self):
        """얼굴 탐지 모델 로드"""
        try:
            # OpenCV Haar Cascade 사용 (가장 안정적)
            self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 얼굴 탐지기가 제대로 로드되었는지 확인
            if self.face_detector.empty():
                print("❌ 얼굴 탐지기 로드 실패 - 빈 분류기")
                self.face_detector = None
                self.use_face_detection = False
                return False
            
            print("✅ 얼굴 탐지기 로드 완료")
            return True
        except Exception as e:
            print(f"❌ 얼굴 탐지기 로드 실패: {e}")
            self.face_detector = None
            self.use_face_detection = False
            return False

    def load_models_with_debugging(self):
        """디버깅을 포함한 모델 로드"""
        self.debug_model_loading()
        
        success_flags = {'person_model': False, 'mask_model': False, 'face_detector': False}
        
        try:
            print("YOLOv8 세그먼테이션 모델 로딩 중...")
            self.person_model = YOLO('yolov8n-seg.pt')
            print("✅ YOLOv8 세그먼테이션 모델 로드 성공")
            success_flags['person_model'] = True
        except Exception as e:
            print(f"❌ YOLOv8 세그먼테이션 모델 로드 실패: {e}")
        
        # 마스크 모델 로드 시도
        mask_model_paths = [
            'mask_best.pt', 
            './mask_best.pt', 
            'models/mask_best.pt',
            'video/mask_best.pt',
            os.path.join(os.getcwd(), 'mask_best.pt')
        ]
        
        for path in mask_model_paths:
            if os.path.exists(path):
                try:
                    print(f"마스크 모델 로딩 시도: {path}")
                    self.mask_model = YOLO(path)
                    print(f"✅ 마스크 모델 로드 성공: {path}")
                    success_flags['mask_model'] = True
                    
                    # 모델 정보 디버깅
                    self.debug_mask_model()
                    break
                    
                except Exception as e:
                    print(f"❌ {path} 로드 실패: {e}")
                    continue
        
        if not success_flags['mask_model']:
            print("⚠️ 마스크 모델을 찾을 수 없습니다. YOLOv8 모델만 사용합니다.")
            self.use_hybrid_approach = False
        else:
            self.use_hybrid_approach = True
        
        # 얼굴 탐지기 로드
        success_flags['face_detector'] = self.load_face_detector()
        
        # GPU 설정
        device = 'cuda' if (torch.cuda.is_available() and self.performance_settings['use_gpu_acceleration']) else 'cpu'
        print(f"디바이스 설정: {device}")
        
        if self.person_model:
            self.person_model.to(device)
        if self.mask_model:
            self.mask_model.to(device)
        
        print(f"모델 로딩 완료 - 하이브리드 모드: {self.use_hybrid_approach}, 얼굴 탐지: {self.use_face_detection}")
        return success_flags

    def load_models(self):
        """레거시 호환용 함수"""
        results = self.load_models_with_debugging()
        return results['person_model'] or results['mask_model']

    def compute_iou(self, boxA, boxB):
        """두 경계 상자 간의 IoU 계산"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        if xA < xB and yA < yB:
            interArea = (xB - xA) * (yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            return interArea / float(boxAArea + boxBArea - interArea)
        return 0.0

    def draw_l_corners(self, frame, x1, y1, x2, y2, color, thickness=4):
        """L형 모서리 브래킷 그리기"""
        corner_len = min(30, (x2-x1)//4, (y2-y1)//4)
        
        # 좌상단 모서리
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thickness)
        
        # 우상단 모서리
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thickness)
        
        # 좌하단 모서리
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thickness)
        
        # 우하단 모서리
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thickness)

    def draw_cursor_pointer(self, frame, x1, y1, x2, y2):
        """마스크 착용자를 위한 다이아몬드 형태의 커서 포인터 그리기"""
        cx = (x1 + x2) // 2
        cy = y1 - 50
        
        diamond_size = 20
        
        top_point = (cx, cy - diamond_size)
        left_point = (cx - diamond_size//2, cy - diamond_size//2)
        bottom_point = (cx, cy)
        right_point = (cx + diamond_size//2, cy - diamond_size//2)
        center_point = (cx, cy - diamond_size//2)
        
        # 메인 다이아몬드 (어두운 녹색)
        main_diamond = np.array([top_point, right_point, bottom_point, left_point], np.int32)
        cv2.fillPoly(frame, [main_diamond], (0, 150, 0))
        
        # 3D 효과를 위한 면들
        top_face = np.array([top_point, left_point, center_point], np.int32)
        cv2.fillPoly(frame, [top_face], (50, 200, 50))
        
        left_face = np.array([left_point, bottom_point, center_point], np.int32)
        cv2.fillPoly(frame, [left_face], (100, 255, 100))
        
        right_face = np.array([top_point, center_point, right_point], np.int32)
        cv2.fillPoly(frame, [right_face], (0, 120, 0))
        
        bottom_face = np.array([center_point, bottom_point, right_point], np.int32)
        cv2.fillPoly(frame, [bottom_face], (0, 100, 0))
        
        # 다이아몬드 윤곽선 및 내부 선들
        cv2.polylines(frame, [main_diamond], True, (255, 255, 255), 2)
        cv2.line(frame, top_point, center_point, (255, 255, 255), 1)
        cv2.line(frame, left_point, center_point, (255, 255, 255), 1)
        cv2.line(frame, right_point, center_point, (255, 255, 255), 1)
        cv2.line(frame, bottom_point, center_point, (255, 255, 255), 1)
        
        # 하이라이트
        highlight_point = (cx - 5, cy - diamond_size + 5)
        cv2.circle(frame, highlight_point, 3, (200, 255, 200), -1)
        
        # 라벨
        label = "MASK"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x = cx - label_size[0] // 2
        label_y = cy - diamond_size - 15
        
        cv2.rectangle(frame, (label_x - 5, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 5, label_y + 5), (0, 150, 0), -1)
        cv2.rectangle(frame, (label_x - 5, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 5, label_y + 5), (255, 255, 255), 1)
        
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def detect_people_and_segment(self, frame, confidence_threshold=0.5):
        """1-2단계: YOLOv8-seg로 사람 탐지 및 세그먼테이션"""
        if self.person_model is None:
            return frame, [], []
        
        try:
            results = self.person_model(frame, classes=[0], conf=confidence_threshold, verbose=False)
            
            detected_people = []
            segmented_people = []
            result_frame = frame.copy()
            
            for result in results:
                # 박스 정보 처리
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        if (x2 - x1) > 50 and (y2 - y1) > 100:
                            detected_people.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': conf
                            })
                            
                            # L형 브래킷 그리기
                            self.draw_l_corners(result_frame, x1, y1, x2, y2, self.person_color)
                
                # 세그먼테이션 마스크 처리
                if hasattr(result, 'masks') and result.masks is not None:
                    boxes = result.boxes
                    masks = result.masks.data
                    
                    for i, (box, mask_tensor) in enumerate(zip(boxes, masks)):
                        seg_conf = float(box.conf[0])
                        if seg_conf < confidence_threshold:
                            continue
                        
                        sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                        
                        # 세그먼테이션 마스크 적용
                        mask_np = mask_tensor.cpu().numpy()
                        frame_h, frame_w = frame.shape[:2]
                        mask_resized = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                        binary_mask = (mask_resized > 0.5).astype(np.uint8)
                        
                        # 색상 오버레이
                        colored_mask = np.zeros_like(frame)
                        colored_mask[binary_mask == 1] = self.segment_color
                        
                        mask_idx = binary_mask == 1
                        if np.any(mask_idx):
                            result_frame[mask_idx] = cv2.addWeighted(
                                result_frame[mask_idx], 0.7, 
                                colored_mask[mask_idx], 0.3, 0
                            )
                        
                        # 윤곽선 그리기
                        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest_contour) > 1000:
                                cv2.drawContours(result_frame, [largest_contour], -1, self.segment_color, 2)
                        
                        # 해당하는 사람 찾기
                        matched_person = None
                        for person in detected_people:
                            iou = self.compute_iou((sx1, sy1, sx2, sy2), person['bbox'])
                            if iou > 0.3:
                                matched_person = person
                                break
                        
                        if matched_person:
                            segmented_people.append({
                                'bbox': matched_person['bbox'],
                                'conf': seg_conf,
                                'mask': binary_mask
                            })
            
            return result_frame, detected_people, segmented_people
            
        except Exception as e:
            print(f"사람 탐지/세그먼테이션 오류: {e}")
            return frame, [], []

    def detect_face_in_person_region(self, frame, person_bbox):
        """사람 영역 내에서 얼굴 탐지"""
        if self.face_detector is None or not self.use_face_detection:
            return None
        
        px1, py1, px2, py2 = person_bbox
        
        # 사람 영역 크롭
        person_crop = frame[py1:py2, px1:px2]
        if person_crop.size == 0:
            return None
        
        # 그레이스케일 변환
        gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 탐지 (사람 영역 내에서)
        faces = self.face_detector.detectMultiScale(
            gray_crop,
            scaleFactor=1.1,
            minNeighbors=3,  # 더 관대하게 설정
            minSize=(20, 20),  # 더 작은 얼굴도 탐지
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None
        
        # 가장 큰 얼굴 선택 (여러 개 탐지된 경우)
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        fx, fy, fw, fh = largest_face
        
        # 원본 프레임 좌표계로 변환
        face_x1 = px1 + fx
        face_y1 = py1 + fy
        face_x2 = px1 + fx + fw
        face_y2 = py1 + fy + fh
        
        # 얼굴 영역을 약간 확장 (마스크까지 포함하도록)
        expand_ratio = 0.4  # 40% 확장
        center_x = (face_x1 + face_x2) // 2
        center_y = (face_y1 + face_y2) // 2
        
        expanded_w = int(fw * (1 + expand_ratio))
        expanded_h = int(fh * (1 + expand_ratio))
        
        expanded_x1 = max(0, center_x - expanded_w // 2)
        expanded_y1 = max(0, center_y - expanded_h // 2)
        expanded_x2 = min(frame.shape[1], center_x + expanded_w // 2)
        expanded_y2 = min(frame.shape[0], center_y + expanded_h // 2)
        
        return {
            'face_bbox': (expanded_x1, expanded_y1, expanded_x2, expanded_y2),
            'original_face_bbox': (face_x1, face_y1, face_x2, face_y2),
            'confidence': 0.8  # Haar Cascade는 신뢰도 미제공
        }

    def classify_mask_detection(self, class_name, class_id):
        """클래스 이름과 ID를 기반으로 마스크 착용 여부 분류"""
        class_name_lower = class_name.lower()
        
        # 마스크 착용 키워드
        mask_keywords = ['mask', 'with_mask', 'face_mask', 'wearing_mask']
        no_mask_keywords = ['no_mask', 'without_mask', 'nomask', 'no-mask']
        
        # 키워드 기반 분류
        for keyword in mask_keywords:
            if keyword in class_name_lower and not any(no_kw in class_name_lower for no_kw in no_mask_keywords):
                return True
        
        for keyword in no_mask_keywords:
            if keyword in class_name_lower:
                return False
        
        # 클래스 ID 기반 분류 (일반적인 패턴)
        # 대부분의 마스크 모델에서 0: mask, 1: no_mask 또는 그 반대
        if class_id == 0:
            return True  # 일반적으로 0번이 마스크
        else:
            return False

    def detect_masks_with_face_detection(self, frame, segmented_people, confidence_threshold=0.3):
        """개선된 마스크 탐지: 사람 영역 → 얼굴 탐지 → 마스크 분류"""
        if self.mask_model is None or not segmented_people:
            print("마스크 모델이 없거나 세그먼테이션된 사람이 없습니다.")
            return frame, [], segmented_people
        
        try:
            print(f"얼굴 탐지 기반 마스크 탐지 시작 - 대상: {len(segmented_people)}명")
            
            mask_wearers = []
            non_mask_people = []
            detected_faces = []  # 탐지된 얼굴 정보 저장
            result_frame = frame.copy()
            
            for i, person in enumerate(segmented_people):
                person_bbox = person['bbox']
                print(f"\n[{i+1}/{len(segmented_people)}] 사람 처리: {person_bbox}")
                
                # 1단계: 사람 영역 내에서 얼굴 탐지
                face_info = None
                if self.use_face_detection:
                    face_info = self.detect_face_in_person_region(frame, person_bbox)
                
                if face_info is None and self.use_face_detection:
                    print("  얼굴을 찾을 수 없어 전체 사람 영역으로 마스크 탐지 시도")
                    # 얼굴이 탐지되지 않으면 기존 방식으로 폴백
                    mask_crop = frame[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]
                    crop_info = {
                        'crop': mask_crop,
                        'bbox': person_bbox,
                        'type': 'person_region'
                    }
                elif face_info is not None:
                    face_bbox = face_info['face_bbox']
                    print(f"  얼굴 탐지 성공: {face_bbox}")
                    
                    # 얼굴 영역 저장
                    detected_faces.append(face_info)
                    
                    # 얼굴 영역 시각화 (시안색 박스)
                    fx1, fy1, fx2, fy2 = face_bbox
                    cv2.rectangle(result_frame, (fx1, fy1), (fx2, fy2), self.face_color, 2)
                    cv2.putText(result_frame, "FACE", (fx1, fy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.face_color, 1)
                    
                    # 2단계: 얼굴 영역 크롭
                    face_crop = frame[fy1:fy2, fx1:fx2]
                    crop_info = {
                        'crop': face_crop,
                        'bbox': face_bbox,
                        'type': 'face_region'
                    }
                else:
                    # 얼굴 탐지를 사용하지 않는 경우 전체 사람 영역 사용
                    mask_crop = frame[person_bbox[1]:person_bbox[3], person_bbox[0]:person_bbox[2]]
                    crop_info = {
                        'crop': mask_crop,
                        'bbox': person_bbox,
                        'type': 'person_region'
                    }
                
                if crop_info['crop'].size == 0:
                    print("  크롭 실패")
                    non_mask_people.append({
                        'bbox': person_bbox,
                        'conf': person['conf'],
                        'class_name': 'crop_failed'
                    })
                    continue
                
                # 3단계: 크롭된 영역에서 마스크 탐지
                print(f"  {crop_info['type']} 크롭 크기: {crop_info['crop'].shape}, 마스크 탐지 시작...")
                
                mask_results = self.mask_model(crop_info['crop'], conf=confidence_threshold, verbose=False)
                
                best_mask_detection = None
                best_no_mask_detection = None
                
                for result in mask_results:
                    if not hasattr(result, 'boxes') or result.boxes is None:
                        continue
                        
                    for box in result.boxes:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        class_name = self.mask_model_classes.get(cls, f'class_{cls}')
                        
                        print(f"    탐지됨: {class_name} (신뢰도: {conf:.3f})")
                        
                        # 마스크 vs 노마스크 분류
                        is_mask = self.classify_mask_detection(class_name, cls)
                        
                        if is_mask:
                            if best_mask_detection is None or conf > best_mask_detection['conf']:
                                best_mask_detection = {
                                    'conf': conf,
                                    'class_name': class_name,
                                    'class': cls
                                }
                        else:
                            if best_no_mask_detection is None or conf > best_no_mask_detection['conf']:
                                best_no_mask_detection = {
                                    'conf': conf,
                                    'class_name': class_name,
                                    'class': cls
                                }
                
                # 4단계: 최종 분류 및 시각화
                if best_mask_detection and (best_no_mask_detection is None or best_mask_detection['conf'] > best_no_mask_detection['conf']):
                    # 마스크 착용자
                    mask_data = {
                        'bbox': person_bbox,
                        'conf': best_mask_detection['conf'],
                        'class_name': best_mask_detection['class_name'],
                        'detection_type': crop_info['type']
                    }
                    
                    if face_info:
                        mask_data['face_bbox'] = face_info['face_bbox']
                    
                    mask_wearers.append(mask_data)
                    
                    # 다이아몬드 커서 표시
                    self.draw_cursor_pointer(result_frame, *person_bbox)
                    
                    print(f"  최종 결정: 마스크 착용 ({best_mask_detection['class_name']}, {best_mask_detection['conf']:.3f}) - {crop_info['type']}")
                    
                else:
                    # 마스크 미착용자
                    detection_info = best_no_mask_detection if best_no_mask_detection else {'conf': person['conf'], 'class_name': 'no_detection'}
                    
                    no_mask_data = {
                        'bbox': person_bbox,
                        'conf': detection_info['conf'],
                        'class_name': detection_info['class_name'],
                        'detection_type': crop_info['type']
                    }
                    
                    if face_info:
                        no_mask_data['face_bbox'] = face_info['face_bbox']
                    
                    non_mask_people.append(no_mask_data)
                    
                    print(f"  최종 결정: 마스크 미착용 ({detection_info['class_name']}, {detection_info['conf']:.3f}) - {crop_info['type']}")
            
            # 얼굴 탐지 결과 저장
            self.detected_faces = detected_faces
            
            print(f"\n최종 결과: 마스크 착용 {len(mask_wearers)}명, 미착용 {len(non_mask_people)}명")
            print(f"얼굴 탐지 성공: {len(detected_faces)}개")
            
            return result_frame, mask_wearers, non_mask_people
            
        except Exception as e:
            print(f"얼굴 탐지 기반 마스크 탐지 오류: {e}")
            import traceback
            traceback.print_exc()
            # 실패 시 기존 방식으로 폴백
            return self.detect_masks_with_model(frame, segmented_people, confidence_threshold)

    def detect_masks_with_model(self, frame, segmented_people, confidence_threshold=0.3):
        """기존 마스크 탐지 방식 (폴백용)"""
        if self.mask_model is None or not segmented_people:
            print("마스크 모델이 없거나 세그먼테이션된 사람이 없습니다.")
            return frame, [], segmented_people
        
        try:
            print(f"기존 방식 마스크 탐지 시작 - 대상: {len(segmented_people)}명")
            
            # 낮은 신뢰도로 마스크 모델 실행하여 더 많은 탐지 결과 확보
            mask_results = self.mask_model(frame, conf=0.1, verbose=False)
            
            mask_wearers = []
            non_mask_people = []
            result_frame = frame.copy()
            
            # 마스크 탐지 결과 수집 및 분류
            all_mask_detections = []
            
            for result in mask_results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    class_name = self.mask_model_classes.get(cls, f'class_{cls}')
                    
                    all_mask_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class': cls,
                        'class_name': class_name
                    })
            
            print(f"마스크 모델 원본 탐지 결과: {len(all_mask_detections)}개")
            
            # 세그먼테이션된 사람들과 매칭
            for person in segmented_people:
                person_bbox = person['bbox']
                px1, py1, px2, py2 = person_bbox
                
                best_mask_match = None
                best_no_mask_match = None
                best_mask_score = 0
                best_no_mask_score = 0
                
                for detection in all_mask_detections:
                    dx1, dy1, dx2, dy2 = detection['bbox']
                    
                    # IoU 계산
                    iou = self.compute_iou(person_bbox, detection['bbox'])
                    
                    # 중심점 거리 계산
                    person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                    detection_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)
                    center_distance = np.sqrt((person_center[0] - detection_center[0])**2 + 
                                            (person_center[1] - detection_center[1])**2)
                    
                    # 얼굴 영역인지 확인 (사람 영역의 상단 40%)
                    face_region_y = py1 + (py2 - py1) * 0.4
                    is_in_face_region = dy1 < face_region_y
                    
                    # 매칭 점수 계산
                    match_score = 0
                    if iou > 0.1:
                        match_score += iou * 0.4
                    
                    person_area = (px2 - px1) * (py2 - py1)
                    max_distance = np.sqrt(person_area) / 2
                    if center_distance < max_distance:
                        distance_score = (max_distance - center_distance) / max_distance
                        match_score += distance_score * 0.3
                    
                    if is_in_face_region:
                        match_score += 0.2
                    
                    match_score += detection['conf'] * 0.1
                    
                    # 매칭 조건 확인
                    if (iou > 0.05 or (center_distance < max_distance * 0.8 and is_in_face_region)) and detection['conf'] > 0.1:
                        is_mask = self.classify_mask_detection(detection['class_name'], detection['class'])
                        
                        if is_mask and match_score > best_mask_score:
                            best_mask_match = detection
                            best_mask_score = match_score
                        elif not is_mask and match_score > best_no_mask_score:
                            best_no_mask_match = detection
                            best_no_mask_score = match_score
                
                # 최종 분류 결정
                if best_mask_match and (best_no_mask_match is None or best_mask_score > best_no_mask_score):
                    mask_wearers.append({
                        'bbox': person_bbox,
                        'conf': best_mask_match['conf'],
                        'class_name': best_mask_match['class_name'],
                        'detection_type': 'legacy_method'
                    })
                    self.draw_cursor_pointer(result_frame, *person_bbox)
                else:
                    chosen_match = best_no_mask_match if best_no_mask_match else {'conf': person['conf'], 'class_name': 'no_detection'}
                    non_mask_people.append({
                        'bbox': person_bbox,
                        'conf': chosen_match['conf'],
                        'class_name': chosen_match['class_name'],
                        'detection_type': 'legacy_method'
                    })
            
            return result_frame, mask_wearers, non_mask_people
            
        except Exception as e:
            print(f"기존 방식 마스크 탐지 오류: {e}")
            # 실패 시 모든 세그먼테이션된 사람을 미착용자로 분류
            non_mask_people = [{'bbox': p['bbox'], 'conf': p['conf'], 'class_name': 'error', 'detection_type': 'error'} for p in segmented_people]
            return frame, [], non_mask_people

    def test_mask_model_only(self, frame, confidence_threshold=0.5):
        """마스크 모델만 단독 테스트"""
        if self.mask_model is None:
            print("마스크 모델이 없습니다.")
            return frame, []
        
        try:
            print("마스크 모델 단독 테스트 시작...")
            results = self.mask_model(frame, conf=confidence_threshold, verbose=False)
            
            detections = []
            result_frame = frame.copy()
            
            for result in results:
                if not hasattr(result, 'boxes') or result.boxes is None:
                    continue
                    
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    class_name = self.mask_model_classes.get(cls, f'class_{cls}')
                    
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'conf': conf,
                        'class': cls,
                        'class_name': class_name
                    })
                    
                    # 탐지 결과 시각화
                    color = (0, 255, 0) if 'mask' in class_name.lower() and 'no' not in class_name.lower() else (255, 0, 0)
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(result_frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(result_frame, label, (x1, y1 - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            print(f"마스크 모델 탐지 결과: {len(detections)}개")
            for det in detections:
                print(f"  - {det['class_name']} (클래스 {det['class']}): {det['conf']:.3f}")
            
            return result_frame, detections
            
        except Exception as e:
            print(f"마스크 모델 테스트 오류: {e}")
            import traceback
            traceback.print_exc()
            return frame, []

    def detect_objects_optimized(self, frame, confidence_threshold=0.5, use_round_robin=True):
        """메인 탐지 함수 - 얼굴 탐지 기반 개선 버전"""
        if self.person_model is None:
            if not self.load_models():
                print("모델 로드 실패")
                return frame, pd.DataFrame()
        
        try:
            # 초기화
            self.detected_people = []
            self.segmented_people = []
            self.mask_wearers = []
            self.non_mask_people = []
            self.detected_faces = []
            
            print("=== 완전 개선된 마스크 탐지 시스템 시작 ===")
            
            # 1-2단계: YOLOv8-seg로 사람 탐지 및 세그먼테이션 (기존과 동일)
            result_frame, detected_people, segmented_people = self.detect_people_and_segment(frame, confidence_threshold)
            
            self.detected_people = detected_people
            self.segmented_people = segmented_people
            
            print(f"1-2단계 완료: {len(detected_people)}명 탐지, {len(segmented_people)}명 세그먼테이션")
            
            if not detected_people:
                print("사람이 탐지되지 않아 마스크 탐지 건너뛰기")
                detections = self.create_detection_dataframe()
                return result_frame, detections
            
            # 3단계: 마스크 탐지 (개선된 방식 또는 기존 방식)
            if self.use_hybrid_approach and self.mask_model is not None:
                if self.use_face_detection and self.face_detector is not None:
                    print("얼굴 탐지 기반 마스크 탐지 사용")
                    result_frame, mask_wearers, non_mask_people = self.detect_masks_with_face_detection(
                        result_frame, segmented_people, confidence_threshold
                    )
                else:
                    print("기존 방식 마스크 탐지 사용")
                    result_frame, mask_wearers, non_mask_people = self.detect_masks_with_model(
                        result_frame, segmented_people, confidence_threshold
                    )
                
                self.mask_wearers = mask_wearers
                self.non_mask_people = non_mask_people
                
                print(f"3단계 완료: {len(mask_wearers)}명 마스크 착용, {len(non_mask_people)}명 미착용")
            else:
                print("마스크 모델 없음 - 모든 세그먼테이션된 사람을 일반 사람으로 분류")
                self.non_mask_people = [{'bbox': p['bbox'], 'conf': p['conf'], 'detection_type': 'no_mask_model'} for p in segmented_people]
            
            # 호환성을 위한 데이터프레임 생성
            detections = self.create_detection_dataframe()
            
            print("=== 완전 개선된 마스크 탐지 시스템 완료 ===")
            print(f"최종 통계: 탐지된 얼굴 {len(self.detected_faces)}개, 마스크 착용 {len(self.mask_wearers)}명, 미착용 {len(self.non_mask_people)}명")
            
            return result_frame, detections
                
        except Exception as e:
            print(f"마스크 탐지 시스템 오류: {e}")
            import traceback
            traceback.print_exc()
            return frame, pd.DataFrame()

    def create_detection_dataframe(self):
        """현재 탐지 상태로부터 pandas 데이터프레임 생성"""
        detections = []
        
        # 마스크 착용자
        for person in self.mask_wearers:
            x1, y1, x2, y2 = person['bbox']
            detections.append({
                'name': 'person_with_mask',
                'confidence': person['conf'],
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                'detection_method': person.get('detection_type', 'unknown')
            })
        
        # 마스크 미착용자
        for person in self.non_mask_people:
            x1, y1, x2, y2 = person['bbox']
            detections.append({
                'name': 'person_no_mask',
                'confidence': person['conf'],
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                'detection_method': person.get('detection_type', 'unknown')
            })
        
        # 세그먼테이션된 사람 (중복 방지를 위해 별도로 추가하지 않음)
        # 대신 통계 정보만 추가
        for person in self.segmented_people:
            x1, y1, x2, y2 = person['bbox']
            detections.append({
                'name': 'person_segmented',
                'confidence': person['conf'],
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2,
                'detection_method': 'segmentation'
            })
        
        return pd.DataFrame(detections)

    def get_system_info(self):
        """시스템 정보 가져오기 - 확장된 버전"""
        return {
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'person_model_loaded': self.person_model is not None,
            'mask_model_loaded': self.mask_model is not None,
            'face_detector_loaded': self.face_detector is not None,
            'use_hybrid_approach': self.use_hybrid_approach,
            'use_face_detection': self.use_face_detection,
            'performance_settings': self.performance_settings,
            'detected_people': len(self.detected_people),
            'segmented_people': len(self.segmented_people),
            'detected_faces': len(self.detected_faces),
            'mask_wearers': len(self.mask_wearers),
            'non_mask_people': len(self.non_mask_people),
            'mask_model_classes': self.mask_model_classes
        }

    def get_frame_at_position(self, video_path, frame_number):
        """특정 위치의 프레임 가져오기"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"비디오 파일을 열 수 없습니다: {video_path}")
                return None
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print(f"프레임 {frame_number}를 읽을 수 없습니다")
                return None
                
            return frame
        except Exception as e:
            print(f"프레임 읽기 오류: {e}")
            return None

    def get_video_info(self, video_path):
        """비디오 정보 가져오기"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0, 0, 0
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            print(f"비디오 정보: {total_frames} 프레임, {fps:.2f} FPS, {duration:.2f}초")
            return fps, total_frames, duration
        except Exception as e:
            print(f"비디오 정보 가져오기 실패: {e}")
            return 0, 0, 0

    def cleanup_resources(self):
        """리소스 정리"""
        self.last_detection_result = None
        self.detected_people = []
        self.segmented_people = []
        self.detected_faces = []
        self.mask_wearers = []
        self.non_mask_people = []
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 전역 비디오 탐지기 인스턴스
video_detector = VideoDetector()

# 초기 모델 로드
print("완전 개선된 마스크 탐지 시스템 초기화 중...")
if not video_detector.load_models():
    print("경고: 일부 모델 로드에 실패했습니다.")
else:
    print("모든 모델이 성공적으로 로드되었습니다.")

print(f"시스템 준비 완료!")
print(f"- 사람 탐지 모델: {'✅' if video_detector.person_model else '❌'}")
print(f"- 마스크 탐지 모델: {'✅' if video_detector.mask_model else '❌'}")
print(f"- 얼굴 탐지 모델: {'✅' if video_detector.face_detector else '❌'}")
print(f"- 하이브리드 모드: {'✅' if video_detector.use_hybrid_approach else '❌'}")
print(f"- 얼굴 탐지 모드: {'✅' if video_detector.use_face_detection else '❌'}")