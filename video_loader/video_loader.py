import streamlit as st
import tempfile
import os
from pathlib import Path
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging
import io
from PIL import Image

# OpenCV 임포트 시도
try:
    import cv2
    OPENCV_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenCV 사용 가능")
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenCV 없음 - 기본 기능만 사용")

# FFmpeg-python 임포트 시도 (대안)
try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
    logger.info("FFmpeg 사용 가능")
except ImportError:
    ffmpeg = None
    FFMPEG_AVAILABLE = False
    logger.warning("FFmpeg 없음")

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class VideoLoader:
    """비디오 파일 로딩 및 관리 핵심 기능"""
    
    def __init__(self):
        self.supported_formats = ["mp4", "mov", "avi", "mkv", "webm", "flv", "wmv"]
        self.max_file_size = 500 * 1024 * 1024  # 500MB 제한
        
    def validate_file(self, uploaded_file) -> Tuple[bool, str]:
        """업로드된 파일 유효성 검사"""
        if uploaded_file is None:
            return False, "파일이 선택되지 않았습니다."
        
        # 파일 확장자 확인
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
        except:
            return False, "파일 확장자를 확인할 수 없습니다."
            
        if file_extension not in self.supported_formats:
            return False, f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(self.supported_formats)}"
        
        # 파일 크기 확인
        if uploaded_file.size > self.max_file_size:
            max_size_mb = self.max_file_size // (1024*1024)
            current_size_mb = uploaded_file.size // (1024*1024)
            return False, f"파일 크기가 너무 큽니다. 현재: {current_size_mb}MB, 최대: {max_size_mb}MB"
        
        return True, "유효한 파일입니다."
    
    def check_environment(self) -> Dict[str, Any]:
        """실행 환경 체크"""
        env_info = {
            "opencv_available": OPENCV_AVAILABLE,
            "ffmpeg_available": FFMPEG_AVAILABLE,
            "pillow_available": True,
            "recommended_action": ""
        }
        
        if OPENCV_AVAILABLE:
            env_info["recommended_action"] = "OpenCV 사용 - 모든 기능 사용 가능"
        elif FFMPEG_AVAILABLE:
            env_info["recommended_action"] = "FFmpeg 사용 - 제한적 기능"
        else:
            env_info["recommended_action"] = "기본 모드 - 파일 정보만 표시"
        
        return env_info
    
    def get_video_info(self, video_bytes: bytes) -> Dict[str, Any]:
        """비디오 파일에서 메타데이터 추출"""
        if OPENCV_AVAILABLE:
            return self._get_video_info_opencv(video_bytes)
        elif FFMPEG_AVAILABLE:
            return self._get_video_info_ffmpeg(video_bytes)
        else:
            return self._get_video_info_basic(video_bytes)
    
    def _get_video_info_opencv(self, video_bytes: bytes) -> Dict[str, Any]:
        """OpenCV를 사용한 비디오 정보 추출"""
        temp_path = None
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            # OpenCV로 비디오 정보 추출
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                return {"error": "비디오를 열 수 없습니다."}
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # FPS가 0이거나 비정상적인 경우 처리
            if fps <= 0:
                fps = 25.0  # 기본값
                logger.warning("FPS 정보를 가져올 수 없어 기본값 25로 설정")
            
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "size_mb": len(video_bytes) / (1024 * 1024),
                "method": "opencv"
            }
            
        except Exception as e:
            logger.error(f"OpenCV 비디오 정보 추출 오류: {str(e)}")
            return {"error": f"OpenCV 비디오 정보 추출 실패: {str(e)}"}
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")
    
    def _get_video_info_ffmpeg(self, video_bytes: bytes) -> Dict[str, Any]:
        """FFmpeg를 사용한 비디오 정보 추출"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            # FFmpeg로 정보 추출
            probe = ffmpeg.probe(temp_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            
            if not video_stream:
                return {"error": "비디오 스트림을 찾을 수 없습니다."}
            
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            
            # FPS 계산
            fps_str = video_stream.get('r_frame_rate', '25/1')
            fps_parts = fps_str.split('/')
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 25.0
            
            # 듀레이션
            duration = float(video_stream.get('duration', 0))
            frame_count = int(duration * fps) if duration > 0 else 0
            
            return {
                "fps": fps,
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "duration": duration,
                "resolution": f"{width}x{height}",
                "size_mb": len(video_bytes) / (1024 * 1024),
                "method": "ffmpeg"
            }
            
        except Exception as e:
            return {"error": f"FFmpeg 비디오 정보 추출 실패: {str(e)}"}
        
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
    
    def _get_video_info_basic(self, video_bytes: bytes) -> Dict[str, Any]:
        """기본적인 파일 정보만 추출 (그래픽 없는 환경용)"""
        try:
            size_mb = len(video_bytes) / (1024 * 1024)
            
            return {
                "fps": 25.0,  # 기본값
                "frame_count": 0,  # 알 수 없음
                "width": 0,  # 알 수 없음
                "height": 0,  # 알 수 없음
                "duration": 0,  # 알 수 없음
                "resolution": "알 수 없음",
                "size_mb": size_mb,
                "method": "basic",
                "warning": "그래픽 환경이 없어 기본 정보만 표시됩니다."
            }
            
        except Exception as e:
            return {"error": f"기본 정보 추출 실패: {str(e)}"}

    def extract_thumbnail(self, video_bytes: bytes, frame_position: float = 0.1) -> Optional[np.ndarray]:
        """비디오에서 썸네일 추출"""
        if OPENCV_AVAILABLE:
            return self._extract_thumbnail_opencv(video_bytes, frame_position)
        elif FFMPEG_AVAILABLE:
            return self._extract_thumbnail_ffmpeg(video_bytes, frame_position)
        else:
            return self._create_placeholder_thumbnail()
    
    def _extract_thumbnail_opencv(self, video_bytes: bytes, frame_position: float = 0.1) -> Optional[np.ndarray]:
        """OpenCV를 사용한 썸네일 추출"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames <= 0:
                logger.warning("유효한 프레임이 없습니다")
                return None
                
            target_frame = max(1, int(total_frames * frame_position))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # 첫 번째 프레임 시도
                cap = cv2.VideoCapture(temp_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret and frame is not None:
                    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    return None
                
        except Exception as e:
            logger.error(f"OpenCV 썸네일 추출 오류: {e}")
            return None
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")
    
    def _extract_thumbnail_ffmpeg(self, video_bytes: bytes, frame_position: float = 0.1) -> Optional[np.ndarray]:
        """FFmpeg를 사용한 썸네일 추출"""
        temp_path = None
        thumb_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            # 썸네일 출력 경로
            thumb_path = temp_path.replace('.mp4', '_thumb.jpg')
            
            # FFmpeg로 썸네일 추출
            (
                ffmpeg
                .input(temp_path, ss=f"{frame_position}")
                .output(thumb_path, vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 이미지 로드
            if os.path.exists(thumb_path):
                with Image.open(thumb_path) as img:
                    return np.array(img)
            else:
                return None
                
        except Exception as e:
            logger.error(f"FFmpeg 썸네일 추출 오류: {e}")
            return None
            
        finally:
            for path in [temp_path, thumb_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass
    
    def _create_placeholder_thumbnail(self) -> np.ndarray:
        """플레이스홀더 썸네일 생성"""
        # 간단한 플레이스홀더 이미지 생성
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 128  # 회색 배경
        
        # 텍스트 영역 (간단하게)
        placeholder[200:280, 200:440] = [200, 200, 200]  # 밝은 회색 박스
        
        return placeholder

    def extract_frame_at_position(self, video_bytes: bytes, frame_number: int) -> Optional[np.ndarray]:
        """특정 프레임 번호의 이미지 추출"""
        if OPENCV_AVAILABLE:
            return self._extract_frame_opencv(video_bytes, frame_number)
        elif FFMPEG_AVAILABLE:
            return self._extract_frame_ffmpeg(video_bytes, frame_number)
        else:
            return self._create_placeholder_thumbnail()
    
    def _extract_frame_opencv(self, video_bytes: bytes, frame_number: int) -> Optional[np.ndarray]:
        """OpenCV를 사용한 특정 프레임 추출"""
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                logger.error("프레임 추출을 위한 비디오 열기 실패")
                return None
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            
            if ret and frame is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                return None
                
        except Exception as e:
            logger.error(f"OpenCV 프레임 추출 오류: {str(e)}")
            return None
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")
    
    def _extract_frame_ffmpeg(self, video_bytes: bytes, frame_number: int) -> Optional[np.ndarray]:
        """FFmpeg를 사용한 특정 프레임 추출"""
        temp_path = None
        frame_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            frame_path = temp_path.replace('.mp4', f'_frame_{frame_number}.jpg')
            
            # FFmpeg로 특정 프레임 추출
            (
                ffmpeg
                .input(temp_path)
                .filter('select', f'gte(n,{frame_number})')
                .output(frame_path, vframes=1, format='image2', vcodec='mjpeg')
                .overwrite_output()
                .run(quiet=True)
            )
            
            if os.path.exists(frame_path):
                with Image.open(frame_path) as img:
                    return np.array(img)
            else:
                return None
                
        except Exception as e:
            logger.error(f"FFmpeg 프레임 추출 오류: {e}")
            return None
            
        finally:
            for path in [temp_path, frame_path]:
                if path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except:
                        pass

    def extract_multiple_frames(self, video_bytes: bytes, frame_numbers: list) -> Dict[int, np.ndarray]:
        """여러 프레임을 한번에 추출 (효율성을 위해)"""
        if not OPENCV_AVAILABLE and not FFMPEG_AVAILABLE:
            return {}
            
        temp_path = None
        frames = {}
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(video_bytes)
                temp_path = tmp_file.name
            
            if OPENCV_AVAILABLE:
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    return frames
                
                sorted_frames = sorted(frame_numbers)
                for frame_num in sorted_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                    ret, frame = cap.read()
                    
                    if ret and frame is not None:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames[frame_num] = rgb_frame
                
                cap.release()
            
            return frames
            
        except Exception as e:
            logger.error(f"다중 프레임 추출 오류: {str(e)}")
            return frames
            
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    logger.warning(f"임시 파일 삭제 실패: {e}")
    
    def save_to_temp(self, video_bytes: bytes, filename: str) -> str:
        """비디오를 임시 파일로 저장하고 경로 반환"""
        try:
            temp_dir = Path(tempfile.gettempdir()) / "video_loader"
            temp_dir.mkdir(exist_ok=True)
            
            # 안전한 파일명 생성
            safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            
            # 확장자가 없으면 .mp4 추가
            if '.' not in safe_filename:
                safe_filename += '.mp4'
                
            temp_path = temp_dir / safe_filename
            
            with open(temp_path, 'wb') as f:
                f.write(video_bytes)
            
            logger.info(f"임시 파일 저장 성공: {temp_path}")
            return str(temp_path)
            
        except Exception as e:
            logger.error(f"임시 파일 저장 실패: {str(e)}")
            raise Exception(f"임시 파일 저장 실패: {str(e)}")
    
    def cleanup_temp_files(self):
        """임시 파일들 정리"""
        try:
            temp_dir = Path(tempfile.gettempdir()) / "video_loader"
            if temp_dir.exists():
                cleaned_count = 0
                for file in temp_dir.glob("*"):
                    try:
                        file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"파일 삭제 실패 {file}: {e}")
                
                logger.info(f"임시 파일 {cleaned_count}개 정리 완료")
                return cleaned_count
                
        except Exception as e:
            logger.error(f"임시 파일 정리 오류: {e}")
            return 0
    
    def get_supported_formats(self) -> list:
        """지원되는 파일 형식 목록 반환"""
        return self.supported_formats.copy()
    
    def format_file_size(self, size_bytes: int) -> str:
        """파일 크기를 읽기 쉬운 형태로 변환"""
        if size_bytes == 0:
            return "0 B"
        
        size_names = ["B", "KB", "MB", "GB"]
        i = 0
        size = float(size_bytes)
        
        while size >= 1024.0 and i < len(size_names) - 1:
            size /= 1024.0
            i += 1
        
        return f"{size:.1f} {size_names[i]}"
    
    def format_duration(self, duration_seconds: float) -> str:
        """재생 시간을 읽기 쉬운 형태로 변환"""
        if duration_seconds <= 0:
            return "00:00"
        
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def test_opencv_installation(self) -> bool:
        """OpenCV 설치 및 작동 테스트"""
        try:
            if OPENCV_AVAILABLE:
                test_array = np.zeros((100, 100, 3), dtype=np.uint8)
                cv2.cvtColor(test_array, cv2.COLOR_BGR2RGB)
                return True
            return False
        except Exception as e:
            logger.error(f"OpenCV 테스트 실패: {e}")
            return False

# 전역 비디오 로더 인스턴스
video_loader = VideoLoader()