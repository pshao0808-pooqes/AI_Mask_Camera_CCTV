import subprocess
import platform
import psutil
import time
import os
import shutil
from pathlib import Path
from PIL import Image
import tempfile
import zipfile
from datetime import datetime
import json
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pandas as pd

class CameraController:
    """카메라 제어 및 파일 관리 핵심 기능 + 객체 탐지"""
    
    def __init__(self):
        self.camera_process = None
        self.save_directory = str(Path.home() / "Downloads" / "CameraCaptures")
        
        # 객체 탐지 관련 초기화
        self.model = None
        self.selected_model = 'yolov8n-seg.pt'
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
        self.last_detection_result = None
        
        # 각 단계별 색상 정의 - video.py와 동일
        self.person_color = (0, 255, 0)      # L자형 모서리용 초록색
        self.segment_color = (255, 165, 0)   # 세그멘테이션용 주황색
        self.mask_color = (0, 0, 255)        # 마스크 탐지용 빨간색
        
        # 실시간 카메라 변수
        self.realtime_cap = None
        
    def get_default_camera_folders(self):
        """운영체제별 기본 카메라 저장 폴더 경로"""
        system = platform.system()
        user_home = Path.home()
        
        folders = []
        
        if system == "Windows":
            folders.extend([
                user_home / "Pictures" / "Camera Roll",
                user_home / "Videos" / "Captures",
                user_home / "Pictures",
                user_home / "Videos"
            ])
        elif system == "Darwin":  # macOS
            folders.extend([
                user_home / "Pictures" / "Photo Booth Library" / "Pictures",
                user_home / "Movies" / "Photo Booth Library" / "Movies",
                user_home / "Pictures",
                user_home / "Movies",
                user_home / "Desktop"
            ])
        elif system == "Linux":
            folders.extend([
                user_home / "Pictures",
                user_home / "Videos",
                user_home / "Desktop",
                user_home / "Documents"
            ])
        
        return [folder for folder in folders if folder.exists()]

    def get_camera_command(self):
        """운영체제별 카메라 앱 실행 명령어"""
        system = platform.system()
        
        if system == "Windows":
            return ["start", "microsoft.windows.camera:", "/B"], True
        elif system == "Darwin":  # macOS
            return ["open", "-a", "Photo Booth"], False
        elif system == "Linux":
            commands_to_try = [
                ["cheese"],
                ["kamoso"], 
                ["guvcview"],
                ["vlc", "v4l2://"]
            ]
            return commands_to_try, False
        else:
            return None, False

    def start_camera(self):
        """시스템 카메라 앱 실행"""
        try:
            commands, use_shell = self.get_camera_command()
            
            if commands is None:
                return False, "지원되지 않는 운영체제입니다."
            
            system = platform.system()
            
            if system == "Linux":
                for cmd in commands:
                    try:
                        if use_shell:
                            process = subprocess.Popen(cmd, shell=True)
                        else:
                            process = subprocess.Popen(cmd)
                        
                        time.sleep(1)
                        if process.poll() is None:
                            self.camera_process = process
                            return True, f"{' '.join(cmd)} 실행 성공!"
                    except (subprocess.SubprocessError, FileNotFoundError):
                        continue
                
                return False, "사용 가능한 카메라 앱을 찾을 수 없습니다."
            else:
                if use_shell:
                    process = subprocess.Popen(commands, shell=True)
                else:
                    process = subprocess.Popen(commands)
                
                self.camera_process = process
                return True, "카메라 앱이 실행되었습니다!"
                
        except Exception as e:
            return False, f"카메라 실행 오류: {str(e)}"

    def stop_camera(self):
        """시스템 카메라 앱 종료"""
        try:
            system = platform.system()
            
            if self.camera_process:
                try:
                    self.camera_process.terminate()
                    self.camera_process = None
                except:
                    pass
            
            camera_processes = []
            
            if system == "Windows":
                process_names = ["WindowsCamera.exe", "Camera.exe"]
            elif system == "Darwin":
                process_names = ["Photo Booth", "Camera"]
            elif system == "Linux":
                process_names = ["cheese", "kamoso", "guvcview"]
            else:
                return False, "지원되지 않는 운영체제입니다."
            
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    if any(name.lower() in proc.info['name'].lower() for name in process_names):
                        camera_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            terminated_count = 0
            for proc in camera_processes:
                try:
                    proc.terminate()
                    terminated_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if terminated_count > 0:
                return True, f"{terminated_count}개의 카메라 앱이 종료되었습니다."
            else:
                return True, "실행 중인 카메라 앱이 없습니다."
                
        except Exception as e:
            return False, f"카메라 종료 오류: {str(e)}"

    def check_camera_status(self):
        """카메라 앱 실행 상태 확인"""
        try:
            system = platform.system()
            
            if system == "Windows":
                process_names = ["WindowsCamera.exe", "Camera.exe"]
            elif system == "Darwin":
                process_names = ["Photo Booth", "Camera"]
            elif system == "Linux":
                process_names = ["cheese", "kamoso", "guvcview"]
            else:
                return False, []
            
            running_cameras = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if any(name.lower() in proc.info['name'].lower() for name in process_names):
                        memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                        running_cameras.append({
                            'name': proc.info['name'],
                            'pid': proc.info['pid'],
                            'memory': f"{memory_mb:.1f} MB"
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return len(running_cameras) > 0, running_cameras
            
        except Exception as e:
            return False, []

    def scan_camera_files(self, folders=None, file_types=None, limit_hours=24):
        """카메라 폴더에서 최근 파일들 스캔"""
        if folders is None:
            folders = self.get_default_camera_folders()
            
        if file_types is None:
            file_types = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.mp4', '.avi', '.mov', '.mkv']
        
        files_found = []
        current_time = time.time()
        time_limit = current_time - (limit_hours * 3600)
        
        for folder in folders:
            if not folder.exists():
                continue
                
            try:
                for file_path in folder.rglob('*'):
                    if (file_path.is_file() and 
                        file_path.suffix.lower() in file_types and
                        file_path.stat().st_mtime > time_limit):
                        
                        file_info = {
                            'path': file_path,
                            'name': file_path.name,
                            'size': file_path.stat().st_size,
                            'modified': file_path.stat().st_mtime,
                            'type': 'image' if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'] else 'video',
                            'folder': folder.name
                        }
                        files_found.append(file_info)
            except (PermissionError, OSError):
                continue
        
        files_found.sort(key=lambda x: x['modified'], reverse=True)
        return files_found

    def format_file_size(self, size_bytes):
        """파일 크기를 읽기 쉬운 형태로 변환"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        return f"{size_bytes:.1f} {size_names[i]}"

    def create_save_directory(self, path):
        """저장 디렉터리 생성"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            self.save_directory = path
            return True, f"저장 폴더가 생성되었습니다: {path}"
        except Exception as e:
            return False, f"폴더 생성 실패: {str(e)}"

    def copy_files_to_destination(self, selected_files, destination_folder, organize_by_date=False):
        """선택된 파일들을 목적지 폴더로 복사"""
        try:
            destination_path = Path(destination_folder)
            if not destination_path.exists():
                destination_path.mkdir(parents=True, exist_ok=True)
            
            copied_files = []
            errors = []
            
            for file_info in selected_files:
                try:
                    source_path = Path(file_info['path'])
                    
                    if organize_by_date:
                        file_date = datetime.fromtimestamp(file_info['modified'])
                        date_folder = destination_path / file_date.strftime("%Y-%m-%d")
                        date_folder.mkdir(exist_ok=True)
                        dest_path = date_folder / source_path.name
                    else:
                        dest_path = destination_path / source_path.name
                    
                    # 동일한 파일명이 있는 경우 번호 추가
                    counter = 1
                    original_dest_path = dest_path
                    while dest_path.exists():
                        stem = original_dest_path.stem
                        suffix = original_dest_path.suffix
                        dest_path = original_dest_path.parent / f"{stem}_{counter}{suffix}"
                        counter += 1
                    
                    shutil.copy2(source_path, dest_path)
                    copied_files.append(dest_path.name)
                    
                except Exception as e:
                    errors.append(f"{file_info['name']}: {str(e)}")
            
            return True, copied_files, errors
            
        except Exception as e:
            return False, [], [f"복사 실패: {str(e)}"]

    def create_zip_archive(self, selected_files, zip_name="camera_files.zip"):
        """선택된 파일들을 ZIP으로 압축"""
        try:
            zip_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(zip_buffer.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_info in selected_files:
                    source_path = Path(file_info['path'])
                    zip_file.write(source_path, source_path.name)
            
            return True, zip_buffer.name
        except Exception as e:
            return False, str(e)

    def get_system_info(self):
        """시스템 정보 반환"""
        system = platform.system()
        return {
            'os': f"{system} {platform.release()}",
            'system': system,
            'supported_apps': self._get_supported_apps(system),
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
        }
    
    def _get_supported_apps(self, system):
        """시스템별 지원 앱 목록"""
        if system == "Windows":
            return ["Windows 카메라 (기본 앱)"]
        elif system == "Darwin":
            return ["Photo Booth (기본 앱)"]
        elif system == "Linux":
            return ["Cheese (권장)", "Kamoso", "Guvcview", "VLC (v4l2)"]
        else:
            return ["지원되지 않는 운영체제"]

    # ===== 객체 탐지 기능 - video.py와 동일한 구현 =====
    
    def load_model(self, model_name="yolov8n-seg.pt"):
        """세그멘테이션 모델 로드"""
        try:
            print(f"모델 로딩 중: {model_name}")
            
            if model_name == "mask_best" and os.path.exists('mask_best.pt'):
                self.model = YOLO('mask_best.pt')
                print("사용자 정의 마스크 모델 로드됨")
            else:
                self.model = YOLO('yolov8n-seg.pt')
                print("YOLOv8 세그멘테이션 모델 로드됨")
            
            device = 'cuda' if (torch.cuda.is_available() and self.performance_settings['use_gpu_acceleration']) else 'cpu'
            self.model.to(device)
            
            if torch.cuda.is_available():
                torch.set_num_threads(8)
                print(f"GPU 사용 중: {torch.cuda.get_device_name()}")
            else:
                print("CPU 모드로 실행")
            
            self.selected_model = model_name
            print(f"모델 로딩 완료: {model_name}")
            return self.model
            
        except Exception as e:
            print(f"모델 로딩 실패: {e}")
            return None

    def compute_iou(self, boxA, boxB):
        """두 경계 상자 간의 IoU(Intersection over Union) 계산"""
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
        """L자형 모서리 브래킷 그리기 - video.py와 동일"""
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

    def detect_people_boxes(self, frame, confidence_threshold=0.6):
        """1단계: 사람을 탐지하고 L자형 모서리 그리기 - video.py와 동일"""
        if self.model is None:
            self.load_model(self.selected_model)
            
        if self.model is None:
            return frame, []
        
        try:
            results = self.model(frame, classes=[0], conf=confidence_threshold, verbose=False)
            people = []
            result_frame = frame.copy()
            
            for res in results:
                if hasattr(res, 'boxes') and res.boxes is not None:
                    for box in res.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        if (x2 - x1) > 50 and (y2 - y1) > 100:
                            people.append({
                                'bbox': (x1, y1, x2, y2),
                                'conf': float(box.conf[0])
                            })
                            
                            self.draw_l_corners(result_frame, x1, y1, x2, y2, self.person_color)
            
            return result_frame, people
            
        except Exception as e:
            print(f"사람 탐지 오류: {e}")
            return frame, []

    def apply_segmentation(self, frame, people_list, confidence_threshold=0.6):
        """2단계: 탐지된 사람에 대해 세그멘테이션 오버레이 적용 - video.py와 동일"""
        if self.model is None or not people_list:
            return frame, []
        
        try:
            results = self.model(frame, classes=[0], conf=confidence_threshold, verbose=False)
            result_frame = frame.copy()
            segmented_people = []
            
            for result in results:
                if not hasattr(result, 'masks') or result.masks is None:
                    continue
                
                boxes = result.boxes
                masks = result.masks.data
                
                for i, (box, mask_tensor) in enumerate(zip(boxes, masks)):
                    seg_conf = float(box.conf[0])
                    if seg_conf < confidence_threshold:
                        continue
                    
                    sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                    
                    # 탐지된 사람과 매칭
                    matched_person = None
                    for person in people_list:
                        iou = self.compute_iou((sx1, sy1, sx2, sy2), person['bbox'])
                        if iou > 0.3:
                            matched_person = person
                            break
                    
                    if matched_person is None:
                        continue
                    
                    # 세그멘테이션 마스크 적용
                    mask_np = mask_tensor.cpu().numpy()
                    frame_h, frame_w = frame.shape[:2]
                    mask_resized = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                    binary_mask = (mask_resized > 0.5).astype(np.uint8)
                    
                    # 색상 오버레이 생성
                    colored_mask = np.zeros_like(frame)
                    colored_mask[binary_mask == 1] = self.segment_color
                    
                    # 투명도를 적용하여 오버레이
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
                    
                    segmented_people.append({
                        'bbox': matched_person['bbox'],
                        'conf': seg_conf,
                        'mask': binary_mask
                    })
            
            return result_frame, segmented_people
            
        except Exception as e:
            print(f"세그멘테이션 오류: {e}")
            return frame, []

    def simple_mask_detection(self, face_region):
        """개선된 마스크 탐지 - 더 엄격한 조건"""
        if face_region.size == 0 or face_region.shape[0] < 30 or face_region.shape[1] < 30:
            return False
        
        try:
            # 얼굴 영역을 더 세분화 (입/코 부분 집중)
            h, w = face_region.shape[:2]
            
            # 입/코 영역만 검사 (얼굴 하단 60%)
            mouth_region = face_region[int(h*0.4):, :]
            
            if mouth_region.size == 0:
                return False
                
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            # 더 엄격한 마스크 색상 범위 (흰색/회색/파란색 마스크)
            mask_ranges = [
                ([0, 0, 180], [180, 50, 255]),      # 흰색/연한 회색 마스크
                ([100, 50, 50], [130, 255, 200]),   # 파란색 마스크 (채도 제한)
                ([0, 0, 100], [180, 30, 180])       # 어두운 회색 마스크
            ]
            
            total_pixels = mouth_region.shape[0] * mouth_region.shape[1]
            mask_pixels = 0
            
            for (lower, upper) in mask_ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)
                mask_pixels += np.sum(mask > 0)
            
            mask_ratio = mask_pixels / total_pixels
            
            # 더 높은 임계값 사용 (50% 이상이어야 마스크로 인정)
            is_mask_detected = mask_ratio > 0.5
            
            # 추가 검증: 에지 검출로 마스크 경계 확인
            if is_mask_detected:
                gray = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_ratio = np.sum(edges > 0) / total_pixels
                
                # 마스크가 있다면 뚜렷한 경계선이 있어야 함
                if edge_ratio < 0.1:  # 경계선이 너무 적으면 마스크가 아닐 가능성
                    is_mask_detected = False
            
            return is_mask_detected
            
        except Exception as e:
            print(f"마스크 탐지 오류: {e}")
            return False

    def detect_masks_and_draw_cursor(self, frame, segmented_people, confidence_threshold=0.65):
        """3단계: 개선된 마스크 탐지 및 커서 그리기"""
        if not segmented_people:
            return frame, []
        
        result_frame = frame.copy()
        mask_wearers = []
        
        for person in segmented_people:
            x1, y1, x2, y2 = person['bbox']
            
            # 얼굴 영역을 더 정확하게 추출 (상단 40%)
            face_h = (y2 - y1) // 2.5  # 얼굴 비율 조정
            face_region = frame[y1:y1+int(face_h), x1:x2]
            
            # 얼굴 영역이 너무 작으면 건너뛰기
            if face_region.shape[0] < 20 or face_region.shape[1] < 20:
                continue
            
            mask_detected = self.simple_mask_detection(face_region)
            
            # 디버그 정보 출력 (개발 중에만 사용)
            if hasattr(self, 'debug_mode') and self.debug_mode:
                h, w = face_region.shape[:2]
                print(f"얼굴 영역 크기: {w}x{h}, 마스크 탐지: {mask_detected}")
            
            if mask_detected:
                mask_wearers.append({
                    'bbox': person['bbox'],
                    'conf': person['conf']
                })
                
                self.draw_cursor_pointer(result_frame, x1, y1, x2, y2)
        
        return result_frame, mask_wearers

    # 디버그 모드 설정을 위한 메서드 추가
    def set_debug_mode(self, enabled=True):
        """디버그 모드 설정"""
        self.debug_mode = enabled

    def draw_cursor_pointer(self, frame, x1, y1, x2, y2):
        """사람 머리 위에 다이아몬드 형태의 커서 포인터 그리기 - video.py와 동일"""
        cx = (x1 + x2) // 2
        cy = y1 - 50
        
        # 다이아몬드 크기 설정
        diamond_size = 20
        
        # 다이아몬드 포인트들 정의
        top_point = (cx, cy - diamond_size)
        left_point = (cx - diamond_size//2, cy - diamond_size//2)
        bottom_point = (cx, cy)
        right_point = (cx + diamond_size//2, cy - diamond_size//2)
        center_point = (cx, cy - diamond_size//2)
        
        # 메인 다이아몬드 (어두운 녹색)
        main_diamond = np.array([top_point, right_point, bottom_point, left_point], np.int32)
        cv2.fillPoly(frame, [main_diamond], (0, 150, 0))  # 어두운 녹색
        
        # 상단 밝은 면 (밝은 녹색)
        top_face = np.array([top_point, left_point, center_point], np.int32)
        cv2.fillPoly(frame, [top_face], (50, 200, 50))  # 중간 녹색
        
        # 좌측 면 (더 밝은 녹색)
        left_face = np.array([left_point, bottom_point, center_point], np.int32)
        cv2.fillPoly(frame, [left_face], (100, 255, 100))  # 밝은 녹색
        
        # 우측 면 (어두운 녹색)
        right_face = np.array([top_point, center_point, right_point], np.int32)
        cv2.fillPoly(frame, [right_face], (0, 120, 0))  # 더 어두운 녹색
        
        # 하단 면 (가장 어두운 녹색)
        bottom_face = np.array([center_point, bottom_point, right_point], np.int32)
        cv2.fillPoly(frame, [bottom_face], (0, 100, 0))  # 가장 어두운 녹색
        
        # 다이아몬드 윤곽선
        cv2.polylines(frame, [main_diamond], True, (255, 255, 255), 2)
        
        # 내부 면 구분선들
        cv2.line(frame, top_point, center_point, (255, 255, 255), 1)
        cv2.line(frame, left_point, center_point, (255, 255, 255), 1)
        cv2.line(frame, right_point, center_point, (255, 255, 255), 1)
        cv2.line(frame, bottom_point, center_point, (255, 255, 255), 1)
        
        # 작은 하이라이트 추가 (3D 효과)
        highlight_size = 3
        highlight_point = (cx - 5, cy - diamond_size + 5)
        cv2.circle(frame, highlight_point, highlight_size, (200, 255, 200), -1)
        
        # 라벨 텍스트
        label = "MASK"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x = cx - label_size[0] // 2
        label_y = cy - diamond_size - 15
        
        # 라벨 배경
        cv2.rectangle(frame, (label_x - 5, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 5, label_y + 5), (0, 150, 0), -1)
        cv2.rectangle(frame, (label_x - 5, label_y - label_size[1] - 5), 
                     (label_x + label_size[0] + 5, label_y + 5), (255, 255, 255), 1)
        
        # 라벨 텍스트
        cv2.putText(frame, label, (label_x, label_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def detect_masks_and_draw_cursor(self, frame, segmented_people, confidence_threshold=0.65):
        """3단계: 마스크를 탐지하고 머리 위에 커서 포인터 그리기 - video.py와 동일"""
        if not segmented_people:
            return frame, []
        
        result_frame = frame.copy()
        mask_wearers = []
        
        for person in segmented_people:
            x1, y1, x2, y2 = person['bbox']
            
            # 얼굴 영역 추출
            face_h = (y2 - y1) // 3
            face_region = frame[y1:y1+face_h, x1:x2]
            
            mask_detected = self.simple_mask_detection(face_region)
            
            if mask_detected:
                mask_wearers.append({
                    'bbox': person['bbox'],
                    'conf': person['conf']
                })
                
                self.draw_cursor_pointer(result_frame, x1, y1, x2, y2)
        
        return result_frame, mask_wearers

    def detect_objects_optimized(self, frame, confidence_threshold=0.5, use_round_robin=True):
        """조건부 순차 처리: 사람 탐지 성공 → 세그멘테이션 → 마스크 탐지 - video.py와 동일"""
        if self.model is None:
            self.load_model(self.selected_model)
        
        if self.model is None:
            print("모델 로딩 실패")
            return frame, pd.DataFrame()
        
        try:
            # 초기화
            self.detected_people = []
            self.segmented_people = []
            self.mask_wearers = []
            
            # 1단계: 사람 탐지 및 L자형 모서리 그리기
            print("1단계: 사람 탐지 시작")
            result_frame, detected_people = self.detect_people_boxes(frame, confidence_threshold)
            self.detected_people = detected_people
            
            if not detected_people:
                print("사람이 탐지되지 않음 - 후속 단계 건너뛰기")
                return result_frame, pd.DataFrame()
            
            print(f"{len(detected_people)}명 탐지됨 - 2단계 진행")
            
            # 2단계: 사람이 탐지된 경우에만 세그멘테이션 적용
            result_frame, segmented_people = self.apply_segmentation(result_frame, detected_people, confidence_threshold)
            self.segmented_people = segmented_people
            
            if not segmented_people:
                print("세그멘테이션 실패 - 마스크 탐지 건너뛰기")
                detections = self.create_detection_dataframe()
                return result_frame, detections
            
            print(f"{len(segmented_people)}명 세그멘테이션됨 - 3단계 진행")
            
            # 3단계: 세그멘테이션이 성공한 경우에만 마스크 탐지 및 커서 그리기
            result_frame, mask_wearers = self.detect_masks_and_draw_cursor(result_frame, segmented_people, confidence_threshold)
            self.mask_wearers = mask_wearers
            
            print(f"{len(mask_wearers)}개 마스크 탐지됨")
            
            # 호환성을 위한 탐지 데이터프레임 생성
            detections = self.create_detection_dataframe()
            
            return result_frame, detections
                
        except Exception as e:
            print(f"탐지 오류: {e}")
            return frame, pd.DataFrame()

    def create_detection_dataframe(self):
        """현재 탐지 상태로부터 pandas 데이터프레임 생성 - video.py와 동일"""
        detections = []
        
        # 사람 탐지 결과 추가
        for person in self.detected_people:
            x1, y1, x2, y2 = person['bbox']
            detections.append({
                'name': 'person',
                'confidence': person['conf'],
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2
            })
        
        # 세그멘테이션된 사람 추가
        for person in self.segmented_people:
            x1, y1, x2, y2 = person['bbox']
            detections.append({
                'name': 'person_segmented',
                'confidence': person['conf'],
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2
            })
        
        # 마스크 착용자 추가
        for person in self.mask_wearers:
            x1, y1, x2, y2 = person['bbox']
            detections.append({
                'name': 'person_with_mask',
                'confidence': person['conf'],
                'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2
            })
        
        return pd.DataFrame(detections)

    # ===== 실시간 카메라 탐지 기능 추가 =====
    
    def check_camera_availability(self, camera_index=0):
        """카메라 사용 가능 여부 확인"""
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                return ret, "카메라 사용 가능"
            else:
                return False, "카메라를 열 수 없습니다"
        except Exception as e:
            return False, f"카메라 확인 오류: {str(e)}"
    
    def get_available_cameras(self):
        """사용 가능한 카메라 목록 반환"""
        available_cameras = []
        
        # 일반적으로 0-4번 카메라 확인
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # 카메라 정보 수집
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    available_cameras.append({
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    })
                cap.release()
        
        return available_cameras
    
    def start_realtime_detection(self, camera_index=0, confidence_threshold=0.5):
        """실시간 카메라 객체 탐지 시작"""
        try:
            cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                return None, "카메라를 열 수 없습니다"
            
            # 카메라 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.realtime_cap = cap
            return cap, "실시간 탐지 시작됨"
            
        except Exception as e:
            return None, f"실시간 탐지 시작 오류: {str(e)}"
    
    def capture_and_detect_frame(self, confidence_threshold=0.5):
        """현재 프레임 캡처 및 객체 탐지 - detect_objects_optimized 사용"""
        if not hasattr(self, 'realtime_cap') or self.realtime_cap is None:
            return None, None, "실시간 카메라가 활성화되지 않았습니다"
        
        try:
            ret, frame = self.realtime_cap.read()
            
            if not ret:
                return None, None, "프레임을 읽을 수 없습니다"
            
            # detect_objects_optimized를 사용하여 순차 처리
            result_img, detections = self.detect_objects_optimized(frame, confidence_threshold)
            
            return frame, result_img, detections
            
        except Exception as e:
            return None, None, f"프레임 캡처 오류: {str(e)}"
    
    def stop_realtime_detection(self):
        """실시간 탐지 중지"""
        try:
            if hasattr(self, 'realtime_cap') and self.realtime_cap is not None:
                self.realtime_cap.release()
                self.realtime_cap = None
                return True, "실시간 탐지가 중지되었습니다"
            else:
                return True, "실시간 탐지가 실행 중이 아닙니다"
        except Exception as e:
            return False, f"실시간 탐지 중지 오류: {str(e)}"
    
    def is_realtime_active(self):
        """실시간 탐지 활성 상태 확인"""
        return hasattr(self, 'realtime_cap') and self.realtime_cap is not None and self.realtime_cap.isOpened()
    
    
    
    def process_streamlit_camera_input(self, camera_bytes, confidence_threshold=0.5):
        """Streamlit camera_input 처리 - detect_objects_optimized 사용"""
        try:
            # bytes를 numpy array로 변환
            nparr = np.frombuffer(camera_bytes.read(), np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, None, "이미지를 디코드할 수 없습니다"
            
            # detect_objects_optimized를 사용하여 순차 처리
            result_img, detections = self.detect_objects_optimized(image, confidence_threshold)
            
            return result_img, detections, "성공"
            
        except Exception as e:
            return None, None, f"Streamlit 카메라 입력 처리 오류: {str(e)}"

    def process_camera_file(self, file_path, confidence_threshold=0.5):
        """카메라로 촬영된 파일에 대한 객체 탐지 처리 - detect_objects_optimized 사용"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # 이미지 처리
                image = cv2.imread(str(file_path))
                if image is None:
                    return None, None, "이미지를 읽을 수 없습니다."
                
                result_img, detections = self.detect_objects_optimized(image, confidence_threshold)
                return result_img, detections, "성공"
                
            elif file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # 비디오 처리 (첫 번째 프레임만)
                cap = cv2.VideoCapture(str(file_path))
                if not cap.isOpened():
                    return None, None, "비디오를 열 수 없습니다."
                
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    return None, None, "비디오 프레임을 읽을 수 없습니다."
                
                result_img, detections = self.detect_objects_optimized(frame, confidence_threshold)
                return result_img, detections, "성공"
            else:
                return None, None, "지원되지 않는 파일 형식입니다."
                
        except Exception as e:
            return None, None, f"파일 처리 오류: {str(e)}"

    # ===== 레거시 함수들 (하위 호환성 유지) =====
    
    def load_lightweight_model(self):
        """CPU용 경량화 모델 로드 - 하위 호환성을 위해 유지"""
        return self.load_model('yolov8n-seg.pt')
    
    def detect_people_fast(self, frame, confidence_threshold=0.6):
        """CPU용 고속 사람 탐지 - 실제로는 detect_people_boxes 호출"""
        return self.detect_people_boxes(frame, confidence_threshold)
    
    def simple_mask_detection_fast(self, frame, people_list):
        """CPU용 간단한 마스크 탐지 - 실제로는 detect_masks_and_draw_cursor 호출"""
        if not people_list:
            return frame, []
        
        # segmented_people을 people_list로 설정하고 마스크 탐지 수행
        self.segmented_people = people_list
        return self.detect_masks_and_draw_cursor(frame, people_list)
    
    def detect_objects_fast(self, frame, confidence_threshold=0.5):
        """CPU 최적화된 고속 객체 탐지 - 실제로는 detect_objects_optimized 호출"""
        return self.detect_objects_optimized(frame, confidence_threshold)
    
    def create_detection_dataframe_fast(self):
        """고속 처리용 데이터프레임 생성 - 실제로는 create_detection_dataframe 호출"""
        return self.create_detection_dataframe()
    
    def process_streamlit_camera_input_fast(self, camera_bytes, confidence_threshold=0.5):
        """Streamlit camera_input 고속 처리 - 실제로는 process_streamlit_camera_input 호출"""
        return self.process_streamlit_camera_input(camera_bytes, confidence_threshold)

    # ===== 추가 유틸리티 함수들 =====
    
    def update_settings(self, **kwargs):
        """설정 업데이트"""
        self.performance_settings.update(kwargs)
        
        if 'model_name' in kwargs and kwargs['model_name'] != self.selected_model:
            self.load_model(kwargs['model_name'])

    def get_detection_info(self):
        """현재 탐지 상태 정보 반환"""
        return {
            'detected_people': len(self.detected_people),
            'segmented_people': len(self.segmented_people),
            'mask_wearers': len(self.mask_wearers),
            'current_model': self.selected_model,
            'performance_settings': self.performance_settings
        }

    def cleanup_resources(self):
        """리소스 정리"""
        # 실시간 카메라 해제
        if hasattr(self, 'realtime_cap') and self.realtime_cap is not None:
            self.realtime_cap.release()
            self.realtime_cap = None
        
        # 탐지 결과 초기화
        self.last_detection_result = None
        self.detected_people = []
        self.segmented_people = []
        self.mask_wearers = []
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 전역 카메라 컨트롤러 인스턴스
camera_controller = CameraController()

# UI 실행을 위한 코드
if __name__ == "__main__":
    try:
        from ui_camera import run_camera_ui
        run_camera_ui()
    except ImportError:
        print("UI 모듈을 찾을 수 없습니다. ui_camera.py 파일이 같은 폴더에 있는지 확인하세요.")
    except Exception as e:
        print(f"UI 실행 중 오류 발생: {e}")