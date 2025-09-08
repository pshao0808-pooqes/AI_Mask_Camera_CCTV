#!/usr/bin/env python3
"""
통합 미디어 처리 시스템 메인 실행 파일 (클래스 기반)
- 카메라 모드: 실시간 카메라 객체 탐지 및 파일 관리
- 비디오 모드: 비디오/이미지 배치 처리 및 분석
- 영상 모니터 모드: 비디오 업로드 및 분석
- CCTV 모드: 실시간 CCTV 교통 분석 시스템
"""

import streamlit as st
import sys
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import subprocess
import threading
import time
import webbrowser
import socket

# 프로젝트 경로 설정
BASE_DIR = Path(__file__).parent
CCTV_BACKEND_DIR = BASE_DIR / "cctv" / "backend"
CCTV_FRONTEND_DIR = BASE_DIR / "cctv" / "frontend"
CAMERA_DIR = BASE_DIR / "camera"
VIDEO_DIR = BASE_DIR / "video" 
VIDEO_LOADER_DIR = BASE_DIR / "video_loader"




class BaseMode(ABC):
    """모든 모드의 기본 클래스"""
    
    def __init__(self, name: str, icon: str, description: str):
        self.name = name
        self.icon = icon
        self.description = description
        self._initialized = False
    
    @abstractmethod
    def check_dependencies(self) -> tuple[bool, str]:
        """각 모드별 의존성 체크"""
        pass
    
    @abstractmethod
    def initialize(self) -> tuple[bool, str]:
        """모드 초기화"""
        pass
    
    @abstractmethod
    def render(self):
        """UI 렌더링"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """리소스 정리"""
        pass
    
    def is_initialized(self) -> bool:
        return self._initialized

class CCTVMode(BaseMode):
    """CCTV 교통 분석 모드 클래스"""
    
    def __init__(self):
        super().__init__(
            name="CCTV 교통 분석",
            icon="🚗",
            description="실시간 CCTV 교통 분석 및 웹 대시보드"
        )
        self.flask_process = None
        self.server_port = 5000
    
    def check_dependencies(self) -> tuple[bool, str]:
        """CCTV 모드 의존성 체크"""
        try:
            # 패키지 체크
            missing_packages = []
            
            try:
                import flask
            except ImportError:
                missing_packages.append("flask")
            
            try:
                import flask_cors
            except ImportError:
                missing_packages.append("flask-cors")
            
            try:
                import flask_socketio
            except ImportError:
                missing_packages.append("flask-socketio")
            
            try:
                import cv2
            except ImportError:
                missing_packages.append("opencv-python")
            
            try:
                import torch
            except ImportError:
                missing_packages.append("torch")
            
            try:
                from ultralytics import YOLO
            except ImportError:
                missing_packages.append("ultralytics")
            
            try:
                import requests
            except ImportError:
                missing_packages.append("requests")
            
            if missing_packages:
                return False, f"누락된 패키지: {', '.join(missing_packages)}"
            
            # CCTV 모듈 파일 존재 확인
            required_files = [
                CCTV_BACKEND_DIR / "app.py",
                CCTV_BACKEND_DIR / "cctv_analyzer.py", 
                CCTV_BACKEND_DIR / "database.py",
                CCTV_FRONTEND_DIR / "dashboard.html",
                CCTV_FRONTEND_DIR / "index.html"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path.relative_to(BASE_DIR)))
            
            if missing_files:
                return False, f"누락된 파일: {', '.join(missing_files)}"
            
            return True, "CCTV 모드 준비 완료"
            
        except Exception as e:
            return False, f"의존성 체크 중 오류: {str(e)}"
    
    def initialize(self) -> tuple[bool, str]:
        """CCTV 모드 초기화"""
        try:
            # BASE_DIR을 경로에 추가
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            
            # CCTV backend 디렉토리도 추가
            if str(CCTV_BACKEND_DIR) not in sys.path:
                sys.path.insert(0, str(CCTV_BACKEND_DIR))
            
            # 포트 사용 가능성 체크
            if not self._is_port_available(self.server_port):
                # 다른 포트 찾기
                for port in range(5001, 5010):
                    if self._is_port_available(port):
                        self.server_port = port
                        break
                else:
                    return False, "사용 가능한 포트를 찾을 수 없습니다"
            
            self._initialized = True
            return True, f"CCTV 모드 초기화 완료 (포트: {self.server_port})"
            
        except Exception as e:
            return False, f"CCTV 모드 초기화 실패: {e}"
    
    def _is_port_available(self, port: int) -> bool:
        """포트 사용 가능 여부 확인"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return True
        except OSError:
            return False
    
    def _start_flask_server(self):
        """Flask 서버를 별도 프로세스로 실행"""
        try:
            app_path = CCTV_BACKEND_DIR / "app.py"
            
            if not app_path.exists():
                st.error(f"app.py 파일을 찾을 수 없습니다: {app_path}")
                return False
            
            # 환경 변수 설정
            env = os.environ.copy()
            env['PYTHONPATH'] = str(BASE_DIR)
            
            # Flask 서버 실행 명령
            cmd = [
                sys.executable,
                str(app_path),
                str(self.server_port)
            ]
            
            st.info(f"Flask 서버 시작 중... (포트: {self.server_port})")
            st.code(f"실행 명령: {' '.join(cmd)}")
            st.code(f"작업 디렉토리: {BASE_DIR}")
            st.code(f"app.py 경로: {app_path}")
            
            self.flask_process = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 서버 시작 확인
            time.sleep(3)
            
            if self.flask_process.poll() is None:
                st.success("Flask 서버가 성공적으로 시작되었습니다!")
                return True
            else:
                # 에러 출력 확인
                stdout, stderr = self.flask_process.communicate()
                error_msg = f"서버 시작 실패\nSTDOUT: {stdout}\nSTDERR: {stderr}"
                st.error(error_msg)
                return False
            
        except Exception as e:
            st.error(f"Flask 서버 시작 실패: {e}")
            return False
    
    def _open_browser(self):
        """브라우저 자동 실행"""
        try:
            time.sleep(1)  # 서버 안정화 대기
            url = f'http://localhost:{self.server_port}'
            webbrowser.open(url)
            st.info(f"브라우저에서 {url}이 열립니다.")
        except Exception as e:
            st.warning(f"브라우저 자동 실행 실패: {e}")
    
    def render(self):
        """CCTV 모드 UI 렌더링"""
        if not self._initialized:
            success, message = self.initialize()
            if not success:
                st.error(f"초기화 실패: {message}")
                return
        
        st.markdown("## 🚗 실시간 CCTV 교통 분석 시스템")
        st.markdown("---")
        
        # 시스템 상태 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if self.flask_process and self.flask_process.poll() is None:
                st.success("🟢 Flask 서버 실행 중")
            else:
                st.error("🔴 Flask 서버 중지됨")
        
        with col2:
            st.info(f"🌐 포트: {self.server_port}")
        
        with col3:
            if self._is_port_available(self.server_port):
                st.warning("⚠️ 서버 대기 중")
            else:
                st.success("✅ 서비스 활성화")
        
        # 자동 상태 새로고침 (서버가 실행 중일 때)
        if self.flask_process and self.flask_process.poll() is None:
            # 5초마다 자동 새로고침
            if st.button("🔄 자동 새로고침", key="auto_refresh"):
                time.sleep(1)
                st.rerun()
        
        # 제어 버튼
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🚀 서버 시작", use_container_width=True):
                if self._start_flask_server():
                    time.sleep(1)
                    st.rerun()
        
        with col2:
            if st.button("🌐 브라우저 열기", use_container_width=True):
                self._open_browser()
        
        with col3:
            if st.button("🔄 상태 새로고침", use_container_width=True):
                st.rerun()
        
        with col4:
            if st.button("⏹️ 서버 중지", use_container_width=True):
                if self.flask_process:
                    self.flask_process.terminate()
                    self.flask_process = None
                    st.info("서버가 중지되었습니다.")
                    st.rerun()
        
        # 서버 로그 표시
        if self.flask_process:
            st.markdown("### 📋 서버 상태")
            
            # 실시간 로그 체크
            if st.button("📄 로그 확인"):
                try:
                    if self.flask_process.poll() is None:
                        # 프로세스가 실행 중인 경우
                        st.success("✅ 서버가 정상적으로 실행 중입니다.")
                        st.info(f"🌐 접속 주소: http://localhost:{self.server_port}")
                    else:
                        # 프로세스가 종료된 경우
                        stdout, stderr = self.flask_process.communicate()
                        st.error("❌ 서버가 종료되었습니다.")
                        
                        if stderr:
                            st.code(f"오류 로그:\n{stderr}")
                        if stdout:
                            st.code(f"출력 로그:\n{stdout}")
                            
                except Exception as e:
                    st.error(f"로그 확인 중 오류: {e}")
        
        # 시스템 정보
        st.markdown("### 📊 시스템 정보")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("""
            **🔧 주요 기능:**
            - 실시간 CCTV 교통 분석
            - 차량 객체 탐지 및 추적  
            - 교통량 통계 및 시각화
            - 웹 기반 대시보드
            - Socket.IO 실시간 통신
            """)
        
        with info_col2:
            st.markdown(f"""
            **🌐 접속 정보:**
            - 메인 페이지: http://localhost:{self.server_port}
            - API 엔드포인트: http://localhost:{self.server_port}/api
            - 웹소켓 연결: 자동 설정됨
            
            **📱 사용 방법:**
            1. '서버 시작' 버튼 클릭
            2. '브라우저 열기'로 웹 인터페이스 접속  
            3. CCTV 목록에서 모니터링할 카메라 선택
            """)
        
        # 고급 설정
        with st.expander("⚙️ 고급 설정"):
            new_port = st.number_input("포트 번호", 
                                     min_value=5000, 
                                     max_value=9999, 
                                     value=self.server_port)
            
            if new_port != self.server_port:
                self.server_port = new_port
                st.info(f"포트가 {new_port}로 변경되었습니다. 서버를 재시작하세요.")
            
            # 파일 경로 정보 표시
            st.markdown("**📁 파일 경로 정보:**")
            st.code(f"CCTV Backend 경로: {CCTV_BACKEND_DIR}")
            st.code(f"CCTV Frontend 경로: {CCTV_FRONTEND_DIR}")
            st.code(f"app.py 경로: {CCTV_BACKEND_DIR / 'app.py'}")
            
            # 파일 존재 여부 확인
            backend_files = {
                "app.py": (CCTV_BACKEND_DIR / "app.py").exists(),
                "cctv_analyzer.py": (CCTV_BACKEND_DIR / "cctv_analyzer.py").exists(),
                "database.py": (CCTV_BACKEND_DIR / "database.py").exists()
            }
            
            frontend_files = {
                "dashboard.html": (CCTV_FRONTEND_DIR / "dashboard.html").exists(),
                "index.html": (CCTV_FRONTEND_DIR / "index.html").exists(),
                "dashboard.css": (CCTV_FRONTEND_DIR / "static" / "css" / "dashboard.css").exists(),
                "style.css": (CCTV_FRONTEND_DIR / "static" / "css" / "style.css").exists(),
                "app.js": (CCTV_FRONTEND_DIR / "static" / "js" / "app.js").exists(),
                "dashboard.js": (CCTV_FRONTEND_DIR / "static" / "js" / "dashboard.js").exists()
            }
            
            st.markdown("**Backend 파일:**")
            for file, exists in backend_files.items():
                st.write(f"• {file}: {'✅' if exists else '❌'}")
            
            st.markdown("**Frontend 파일:**")
            for file, exists in frontend_files.items():
                st.write(f"• {file}: {'✅' if exists else '❌'}")
    
    def cleanup(self):
        """CCTV 모드 리소스 정리"""
        if self.flask_process:
            try:
                self.flask_process.terminate()
                self.flask_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flask_process.kill()
            except:
                pass
            self.flask_process = None

class CameraMode(BaseMode):
    """카메라 모드 클래스"""
    
    def __init__(self):
        super().__init__(
            name="카메라 모드",
            icon="📹",
            description="실시간 객체 탐지 및 카메라 파일 관리"
        )
        self.camera_controller = None
        self.ui_components = None
    
    def check_dependencies(self) -> tuple[bool, str]:
        """카메라 모드 의존성 체크"""
        try:
            missing_packages = []
            
            try:
                import cv2
            except ImportError:
                missing_packages.append("opencv-python")
            
            try:
                import torch
            except ImportError:
                missing_packages.append("torch")
            
            try:
                from ultralytics import YOLO
            except ImportError:
                missing_packages.append("ultralytics")
            
            try:
                import psutil
            except ImportError:
                missing_packages.append("psutil")
            
            if missing_packages:
                return False, f"누락된 패키지: {', '.join(missing_packages)}"
            
            # 카메라 모듈 파일 존재 확인
            required_files = [
                CAMERA_DIR / "camera.py",
                CAMERA_DIR / "ui_camera.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path.relative_to(BASE_DIR)))
            
            if missing_files:
                return False, f"누락된 파일: {', '.join(missing_files)}"
            
            return True, "카메라 모드 준비 완료"
            
        except Exception as e:
            return False, f"의존성 체크 중 오류: {str(e)}"
    
    def initialize(self) -> tuple[bool, str]:
        """카메라 모드 초기화"""
        try:
            # 카메라 디렉토리를 경로에 추가
            if str(CAMERA_DIR) not in sys.path:
                sys.path.insert(0, str(CAMERA_DIR))
            
            from camera import CameraController
            from ui_camera import (
                init_session_state, render_camera_control_tab,
                render_file_management_tab, render_unified_detection_tab,
                render_detection_analytics, render_settings_tab, render_sidebar
            )
            
            self.camera_controller = CameraController()
            self.ui_components = {
                'init_session_state': init_session_state,
                'render_camera_control_tab': render_camera_control_tab,
                'render_file_management_tab': render_file_management_tab,
                'render_unified_detection_tab': render_unified_detection_tab,
                'render_detection_analytics': render_detection_analytics,
                'render_settings_tab': render_settings_tab,
                'render_sidebar': render_sidebar
            }
            
            self._initialized = True
            return True, "카메라 모드 초기화 완료"
            
        except Exception as e:
            return False, f"카메라 모드 초기화 실패: {e}"
    
    def render(self):
        """카메라 모드 UI 렌더링"""
        if not self._initialized:
            success, message = self.initialize()
            if not success:
                st.error(f"초기화 실패: {message}")
                return
        
        # 세션 상태 초기화
        self.ui_components['init_session_state']()
        
        # 탭 기반 UI
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎮 카메라 제어", "📁 파일 관리", "🔬 객체 탐지", "📊 분석 결과", "⚙️ 설정"
        ])
        
        with tab1:
            self.ui_components['render_camera_control_tab']()
        with tab2:
            self.ui_components['render_file_management_tab']()
        with tab3:
            self.ui_components['render_unified_detection_tab']()
        with tab4:
            self.ui_components['render_detection_analytics']()
        with tab5:
            self.ui_components['render_settings_tab']()
        
        # 사이드바
        self.ui_components['render_sidebar']()
    
    def cleanup(self):
        """카메라 모드 리소스 정리"""
        if self.camera_controller:
            self.camera_controller.cleanup_resources()

class VideoMode(BaseMode):
    """비디오 모드 클래스"""
    
    def __init__(self):
        super().__init__(
            name="비디오 모드",
            icon="🎬", 
            description="배치 영상 분석 및 고급 분석 차트"
        )
        self.video_detector = None
        self.ui_instance = None
    
    def check_dependencies(self) -> tuple[bool, str]:
        """비디오 모드 의존성 체크"""
        try:
            missing_packages = []
            
            try:
                import cv2
            except ImportError:
                missing_packages.append("opencv-python")
            
            try:
                import torch
            except ImportError:
                missing_packages.append("torch")
            
            try:
                from ultralytics import YOLO
            except ImportError:
                missing_packages.append("ultralytics")
            
            if missing_packages:
                return False, f"누락된 패키지: {', '.join(missing_packages)}"
            
            # 비디오 모듈 파일 존재 확인
            required_files = [
                VIDEO_DIR / "video.py",
                VIDEO_DIR / "ui_video.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path.relative_to(BASE_DIR)))
            
            if missing_files:
                return False, f"누락된 파일: {', '.join(missing_files)}"
            
            return True, "비디오 모드 준비 완료"
            
        except Exception as e:
            return False, f"의존성 체크 중 오류: {str(e)}"
    
    def initialize(self) -> tuple[bool, str]:
        """비디오 모드 초기화"""
        try:
            # 비디오 디렉토리를 경로에 추가
            if str(VIDEO_DIR) not in sys.path:
                sys.path.insert(0, str(VIDEO_DIR))
            
            from video import VideoDetector
            from ui_video import PremiumUI
            
            self.video_detector = VideoDetector()
            self.ui_instance = PremiumUI()
            
            self._initialized = True
            return True, "비디오 모드 초기화 완료"
            
        except Exception as e:
            return False, f"비디오 모드 초기화 실패: {e}"
    
    def render(self):
        """비디오 모드 UI 렌더링"""
        if not self._initialized:
            success, message = self.initialize()
            if not success:
                st.error(f"초기화 실패: {message}")
                return
        
        # UI 렌더링 (헤더 제외)
        settings = self.ui_instance.render_advanced_sidebar()
        self.ui_instance.render_processing_pipeline_visual()
        
        tab1, tab2, tab3 = st.tabs(["이미지 분석", "비디오 분석", "분석 결과"])
        
        with tab1:
            self.ui_instance.render_image_detection_mode(settings)
        with tab2:
            self.ui_instance.render_video_analysis_mode(settings)
        with tab3:
            self.ui_instance.render_analytics_dashboard()
    
    def cleanup(self):
        """비디오 모드 리소스 정리"""
        if self.video_detector:
            self.video_detector.cleanup_resources()

class MonitorMode(BaseMode):
    """영상 모니터 모드 클래스"""
    
    def __init__(self):
        super().__init__(
            name="영상 모니터",
            icon="🎥",
            description="영상 업로드 관리 및 품질 평가"
        )
        self.video_loader = None
        self.ui_components = None
    
    def check_dependencies(self) -> tuple[bool, str]:
        """영상 모니터 모드 의존성 체크"""
        try:
            missing_packages = []
            
            try:
                import cv2
            except ImportError:
                missing_packages.append("opencv-python")
            
            try:
                from PIL import Image
            except ImportError:
                missing_packages.append("pillow")
            
            if missing_packages:
                return False, f"누락된 패키지: {', '.join(missing_packages)}"
            
            # 영상 모니터 모듈 파일 존재 확인
            required_files = [
                VIDEO_LOADER_DIR / "video_loader.py",
                VIDEO_LOADER_DIR / "ui_video_loader.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not file_path.exists():
                    missing_files.append(str(file_path.relative_to(BASE_DIR)))
            
            if missing_files:
                return False, f"누락된 파일: {', '.join(missing_files)}"
            
            return True, "영상 모니터 모드 준비 완료"
            
        except Exception as e:
            return False, f"의존성 체크 중 오류: {str(e)}"
    
    def initialize(self) -> tuple[bool, str]:
        """영상 모니터 모드 초기화"""
        try:
            # 영상 모니터 디렉토리를 경로에 추가
            if str(VIDEO_LOADER_DIR) not in sys.path:
                sys.path.insert(0, str(VIDEO_LOADER_DIR))
            
            from video_loader import VideoLoader
            from ui_video_loader import (
                init_session_state, render_upload_sidebar,
                render_upload_history, render_main_monitor, render_settings
            )
            
            self.video_loader = VideoLoader()
            self.ui_components = {
                'init_session_state': init_session_state,
                'render_upload_sidebar': render_upload_sidebar,
                'render_upload_history': render_upload_history,
                'render_main_monitor': render_main_monitor,
                'render_settings': render_settings
            }
            
            self._initialized = True
            return True, "영상 모니터 모드 초기화 완료"
            
        except Exception as e:
            return False, f"영상 모니터 모드 초기화 실패: {e}"
    
    def render(self):
        """영상 모니터 모드 UI 렌더링"""
        if not self._initialized:
            success, message = self.initialize()
            if not success:
                st.error(f"초기화 실패: {message}")
                return
        
        # 세션 상태 초기화
        self.ui_components['init_session_state']()
        
        # UI 컴포넌트 렌더링
        self.ui_components['render_upload_sidebar']()
        self.ui_components['render_upload_history']()
        self.ui_components['render_main_monitor']()
        self.ui_components['render_settings']()
    
    def cleanup(self):
        """영상 모니터 모드 리소스 정리"""
        if self.video_loader:
            self.video_loader.cleanup_temp_files()

class MediaProcessingSystem:
    """통합 미디어 처리 시스템 메인 클래스"""
    
    def __init__(self):
        self.modes = {
            'camera': CameraMode(),
            'video': VideoMode(), 
            'monitor': MonitorMode(),
            'cctv': CCTVMode()
        }
        self.current_mode: Optional[BaseMode] = None
        self._system_initialized = False
    
    def check_system_dependencies(self) -> tuple[bool, str]:
        """시스템 전체 의존성 체크"""
        try:
            required_packages = [
                'streamlit', 'torch', 'opencv-python', 'numpy', 'pillow',
                'ultralytics', 'pandas', 'plotly', 'psutil'
            ]
            
            missing = []
            for package in required_packages:
                try:
                    if package == 'opencv-python':
                        import cv2
                    elif package == 'pillow':
                        from PIL import Image
                    else:
                        __import__(package.replace('-', '_'))
                except ImportError:
                    missing.append(package)
            
            if missing:
                return False, f"다음 패키지들을 설치해주세요: {', '.join(missing)}"
            
            return True, "시스템 의존성 체크 완료"
            
        except Exception as e:
            return False, f"시스템 체크 중 오류: {e}"
    
    def initialize_session(self):
        """세션 상태 초기화"""
        if 'mode' not in st.session_state:
            st.session_state.mode = None
        if 'system_checked' not in st.session_state:
            st.session_state.system_checked = False
    
    def show_main_menu(self):
        """메인 메뉴 화면 표시"""
        st.title("🤖 통합 미디어 처리 시스템")
        st.markdown("---")
        
        # 모드 선택 카드
        cols = st.columns(len(self.modes))
        
        for i, (mode_key, mode_obj) in enumerate(self.modes.items()):
            with cols[i]:
                st.subheader(f"{mode_obj.icon} {mode_obj.name}")
                
                # 기능 설명
                description_lines = mode_obj.description.split(' 및 ')
                for line in description_lines:
                    st.write(f"• {line}")
                
                # 의존성 체크 상태 표시
                dep_ok, dep_msg = mode_obj.check_dependencies()
                if dep_ok:
                    st.success("✅ 준비 완료")
                else:
                    st.error("❌ 요구사항 미충족")
                    with st.expander("상세 정보"):
                        st.write(dep_msg)
                
                # 모드 시작 버튼
                if st.button(
                    f"{mode_obj.name} 시작",
                    key=mode_key,
                    use_container_width=True,
                    disabled=not dep_ok
                ):
                    st.session_state.mode = mode_key
                    st.rerun()
    
    def show_return_button(self):
        """메인 메뉴 돌아가기 버튼"""
        st.markdown("---")
        if st.button("🏠 메인 메뉴로 돌아가기"):
            # 현재 모드 정리
            if self.current_mode:
                self.current_mode.cleanup()
            
            # 세션 상태 리셋
            st.session_state.mode = None
            self.current_mode = None
            st.rerun()
    
    def run_selected_mode(self, mode_key: str):
        """선택된 모드 실행"""
        if mode_key not in self.modes:
            st.error(f"알 수 없는 모드: {mode_key}")
            return
        
        mode_obj = self.modes[mode_key]
        self.current_mode = mode_obj
        
        # 제목과 돌아가기 버튼
        col1, col2 = st.columns([5, 1])
        with col1:
            st.title(f"{mode_obj.icon} {mode_obj.name}")
        with col2:
            if st.button("🏠", help="메인 메뉴", key=f"{mode_key}_home"):
                st.session_state.mode = None
                st.rerun()
        
        st.markdown("---")
        
        try:
            # 모드 렌더링
            mode_obj.render()
            self.show_return_button()
            
        except Exception as e:
            st.error(f"{mode_obj.name} 실행 중 오류: {e}")
            self.show_return_button()
    
    def run(self):
        """메인 시스템 실행"""
        # 페이지 설정
        st.set_page_config(
            page_title="통합 미디어 처리 시스템",
            page_icon="🤖",
            layout="wide"
        )
        
        # 세션 초기화
        self.initialize_session()
        
        # 시스템 체크 (한 번만)
        if not st.session_state.system_checked:
            success, message = self.check_system_dependencies()
            if not success:
                st.error(message)
                st.code("pip install streamlit torch opencv-python numpy pillow ultralytics pandas plotly psutil flask flask-cors flask-socketio requests")
                return
            st.session_state.system_checked = True
        
        # 모드 실행
        selected_mode = st.session_state.get('mode')
        
        if selected_mode and selected_mode in self.modes:
            self.run_selected_mode(selected_mode)
        else:
            self.show_main_menu()

def main():
    """메인 실행 함수"""
    try:
        system = MediaProcessingSystem()
        system.run()
    except KeyboardInterrupt:
        st.info("프로그램이 중단되었습니다.")
    except Exception as e:
        st.error(f"시스템 오류: {e}")
        st.info("페이지를 새로고침해주세요.")

if __name__ == "__main__":
    main()