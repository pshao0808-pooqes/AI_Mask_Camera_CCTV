import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import threading
import asyncio

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 고급 스타일 CSS
CUSTOM_CSS = """
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-active { background-color: #4CAF50; }
    .status-inactive { background-color: #f44336; }
    .status-processing { background-color: #ff9800; }
    
    .stage-card {
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        background: #f8f9fa;
        border-radius: 0 8px 8px 0;
    }
    
    .processing-pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .pipeline-step {
        text-align: center;
        flex: 1;
        position: relative;
    }
    
    .pipeline-arrow {
        font-size: 24px;
        color: #667eea;
        margin: 0 10px;
    }
</style>
"""

class PremiumUI:
    def __init__(self):
        self.init_page_config()
        self.init_session_state()
        self.load_custom_css()


    def run_standalone(self):
        """독립 실행용 비디오 UI - main.py에서 호출하지 않을 때 사용"""
        # 메인 헤더 렌더링
        self.render_header()
        
        # 사이드바에서 설정값들 가져오기
        settings = self.render_advanced_sidebar()
        
        # 처리 파이프라인 시각화 렌더링
        self.render_processing_pipeline_visual()
        
        # 메인 탭 구성 (이미지 분석, 비디오 분석, 실시간 분석)
        tab1, tab2, tab3 = st.tabs(["이미지 분석", "비디오 분석", "실시간 분석"])
        
        with tab1:
            self.render_image_detection_mode(settings)
        
        with tab2:
            self.render_video_analysis_mode(settings)
        
        with tab3:
            self.render_analytics_dashboard()

    def run(self):
        """메인 UI 실행 함수 - 하위 호환성 유지"""
        self.run_standalone()
        
    def init_page_config(self):
        """페이지 구성 설정"""
        st.set_page_config(
            page_title="고급 YOLO 탐지 시스템",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def init_session_state(self):
        """세션 상태 초기화"""
        defaults = {
            'video_path': None,
            'current_frame': 0,
            'total_frames': 0,
            'fps': 30,
            'is_playing': False,
            'playback_speed': 1.0,
            'last_detection_time': None,
            'processing_history': [],
            'processing_times': [],
            'performance_metrics': {
                'total_processed': 0,
                'avg_processing_time': 0,
                'detection_accuracy': 0
            },
            'video_playing_thread': None,
            'stop_video': False,
            'batch_results': None,
            'processing_complete': False,
            'is_processing': False,
            'auto_show_results': True  # 자동 결과 표시 플래그
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def load_custom_css(self):
        """커스텀 CSS 스타일 로드"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    def render_header(self):
        """메인 헤더 렌더링"""
        st.markdown("""
        <div class="main-header">
            <h1>🔬 고급 순차 YOLO 탐지 시스템</h1>
            <p>전문가급 객체 탐지 및 순차 처리 파이프라인</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_processing_pipeline_visual(self):
        """처리 파이프라인 시각화"""
        from video import video_detector
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            people_count = len(video_detector.detected_people)
            status = "active" if people_count > 0 else "inactive"
            st.markdown(f"""
            <div class="stage-card">
                <h4><span class="status-indicator status-{status}"></span>1단계: 사람 탐지</h4>
                <p>L자형 경계상자</p>
                <strong>{people_count}명 탐지됨</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            seg_count = len(video_detector.segmented_people)
            status = "active" if seg_count > 0 else "inactive"
            st.markdown(f"""
            <div class="stage-card">
                <h4><span class="status-indicator status-{status}"></span>2단계: 세그멘테이션</h4>
                <p>픽셀 단위 분할 마스크</p>
                <strong>{seg_count}명 분할됨</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mask_count = len(video_detector.mask_wearers)
            status = "active" if mask_count > 0 else "inactive"
            st.markdown(f"""
            <div class="stage-card">
                <h4><span class="status-indicator status-{status}"></span>3단계: 마스크 탐지</h4>
                <p>커서 표시기로 마스크 표시</p>
                <strong>{mask_count}개 마스크 탐지됨</strong>
            </div>
            """, unsafe_allow_html=True)
    
    def render_advanced_sidebar(self):
        """고급 사이드바 렌더링"""
        with st.sidebar:
            st.markdown("## 제어 패널")
            
            with st.expander("모델 구성", expanded=True):
                model_choice = st.selectbox(
                    "탐지 모델",
                    ["yolov8n-seg.pt", "mask_best"],
                    help="세그멘테이션 모델 선택",
                    key="sidebar_model_choice"
                )
                
                device_info = "CPU 모드" if not torch.cuda.is_available() else f"GPU: {torch.cuda.get_device_name()}"
                st.info(device_info)
            
            with st.expander("성능 설정", expanded=True):
                detection_interval = st.slider(
                    "탐지 간격",
                    min_value=1, max_value=10, value=3,
                    help="N 프레임마다 처리",
                    key="sidebar_detection_interval"
                )
                
                confidence_threshold = st.slider(
                    "신뢰도 임계값",
                    min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                    help="탐지를 위한 최소 신뢰도",
                    key="sidebar_confidence_threshold"
                )
                
                target_fps = st.slider(
                    "목표 FPS",
                    min_value=15, max_value=60, value=30,
                    help="목표 초당 프레임 수",
                    key="sidebar_target_fps"
                )
            
            with st.expander("성능 메트릭", expanded=False):
                self.render_performance_metrics()
            
            with st.expander("고급 옵션", expanded=False):
                enable_history = st.checkbox("처리 히스토리 활성화", value=True, key="enable_history")
                enable_analytics = st.checkbox("분석 활성화", value=True, key="enable_analytics")
                auto_save_results = st.checkbox("결과 자동 저장", value=False, key="auto_save_results")
            
            return {
                'model_choice': model_choice,
                'detection_interval': detection_interval,
                'confidence_threshold': confidence_threshold,
                'target_fps': target_fps,
                'enable_history': enable_history,
                'enable_analytics': enable_analytics,
                'auto_save_results': auto_save_results
            }
    
    def render_performance_metrics(self):
        """성능 메트릭 표시"""
        metrics = st.session_state.performance_metrics
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("처리됨", metrics['total_processed'])
        with col2:
            st.metric("평균 시간", f"{metrics['avg_processing_time']:.3f}초")
    
    def render_image_detection_mode(self, settings):
        """이미지 탐지 모드 렌더링"""
        st.markdown("## 전문 이미지 분석")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### 분석 설정")
            confidence = st.slider(
                "분석 신뢰도",
                0.1, 0.9, settings['confidence_threshold'], 0.1,
                key="image_analysis_confidence"
            )
            
            show_heatmap = st.checkbox("신뢰도 히트맵 표시", value=False, key="show_heatmap")
            detailed_report = st.checkbox("상세 보고서 생성", value=True, key="detailed_report")
            
        with col1:
            uploaded_file = st.file_uploader(
                "분석할 이미지 업로드",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                help="지원 형식: JPG, PNG, BMP, WEBP",
                key="image_uploader"
            )
            
            if uploaded_file is not None:
                self.process_uploaded_image(uploaded_file, confidence, show_heatmap, detailed_report)
    
    def process_uploaded_image(self, uploaded_file, confidence, show_heatmap, detailed_report):
        """업로드된 이미지 처리 및 분석"""
        from video import video_detector
        
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 처리 결과")
            
            start_time = time.time()
            
            with st.spinner("순차 처리 파이프라인 실행 중..."):
                result_img, detections = video_detector.detect_objects_optimized(img_array, confidence)
                
            processing_time = time.time() - start_time
            
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_container_width=True)
            
            st.success(f"✅ {processing_time:.3f}초 만에 처리 완료")
            
            self.update_performance_metrics(processing_time, len(detections))
        
        with col2:
            st.markdown("### 분석 보고서")
            
            stages_data = {
                '단계': ['사람 탐지', '세그멘테이션', '마스크 탐지'],
                '개수': [
                    len(video_detector.detected_people),
                    len(video_detector.segmented_people),
                    len(video_detector.mask_wearers)
                ],
                '상태': ['✅' if count > 0 else '❌' for count in [
                    len(video_detector.detected_people),
                    len(video_detector.segmented_people),
                    len(video_detector.mask_wearers)
                ]]
            }
            
            df = pd.DataFrame(stages_data)
            st.dataframe(df, use_container_width=True)
            
            if detailed_report and len(detections) > 0:
                st.markdown("### 상세 분석")
                
                if len(detections) > 0:
                    fig = px.histogram(
                        detections, x='confidence',
                        title="신뢰도 분포",
                        nbins=20
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True, key="image_confidence_histogram")
    
    def video_playback_thread(self, video_path, target_fps, enable_detection, confidence):
        """비디오 재생을 위한 백그라운드 스레드 함수"""
        delay = 1.0 / (target_fps * st.session_state.playback_speed)
        
        while st.session_state.is_playing and not st.session_state.stop_video:
            if st.session_state.current_frame >= st.session_state.total_frames - 1:
                st.session_state.is_playing = False
                break
                
            st.session_state.current_frame += 1
            time.sleep(delay)
    
    def render_video_analysis_mode(self, settings):
        """비디오 분석 모드 렌더링"""
        st.markdown("## 전문 비디오 분석")
        
        uploaded_video = st.file_uploader(
            "분석할 비디오 업로드",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="지원 형식: MP4, AVI, MOV, MKV",
            key="video_uploader"
        )
        
        if uploaded_video is not None:
            self.process_uploaded_video_batch(uploaded_video, settings)
    
    def process_uploaded_video_batch(self, uploaded_video, settings):
        """배치 처리로 비디오 분석"""
        from video import video_detector
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(uploaded_video.read())
            video_path = tmp.name
        
        fps, total_frames, duration = video_detector.get_video_info(video_path)
        
        if total_frames == 0:
            st.error("비디오를 읽을 수 없습니다. 다른 파일을 선택해주세요.")
            return
        
        # 비디오 정보 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 프레임", f"{total_frames:,}")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("길이", f"{duration:.1f}초")
        with col4:
            estimated_time = min(total_frames * 0.2, 500)
            st.metric("예상 처리시간", f"{estimated_time:.1f}초")
        
        # 처리 완료 후 자동 결과 표시
        if st.session_state.get('processing_complete', False) and st.session_state.get('batch_results'):
            if st.session_state.get('auto_show_results', True):
                st.success("처리 완료! 결과를 표시합니다.")
                self.display_batch_results(
                    st.session_state['batch_results'], 
                    [r['processing_time'] for r in st.session_state['batch_results']]
                )
                return
            else:
                st.success("이전 처리 결과가 있습니다!")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("이전 결과 보기", type="primary", key="show_prev_results"):
                        st.session_state['auto_show_results'] = True
                        st.rerun()
                with col2:
                    if st.button("새로 처리하기", key="process_new"):
                        st.session_state['processing_complete'] = False
                        st.session_state['batch_results'] = None
                        st.session_state['auto_show_results'] = True
                        st.rerun()
                return
        
        if not st.session_state.get('is_processing', False):
            st.markdown("### 처리 옵션")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                process_interval = st.selectbox(
                    "처리 간격",
                    [1, 2, 3, 5, 10],
                    index=2,
                    help="N 프레임마다 처리 (간격이 클수록 빨라짐)",
                    key="process_interval_select"
                )
            
            with col2:
                # 최대 프레임 제한을 대폭 확장하고 "전체" 옵션 추가
                max_frames_options = [100, 500, 1000, 2000, 5000, total_frames]
                max_frames_labels = ["100개", "500개", "1000개", "2000개", "5000개", f"전체 ({total_frames}개)"]
                
                max_frames_idx = st.selectbox(
                    "최대 처리 프레임",
                    range(len(max_frames_options)),
                    index=2,  # 기본값을 1000개로 설정
                    format_func=lambda x: max_frames_labels[x],
                    help="처리할 최대 프레임 수",
                    key="max_frames_select"
                )
                max_frames = max_frames_options[max_frames_idx]
            
            with col3:
                confidence = st.slider(
                    "신뢰도 임계값",
                    0.3, 0.8, 0.5, 0.1,
                    help="탐지 신뢰도 설정",
                    key="confidence_select"
                )
            
            # 예상 처리 시간 계산
            frames_to_process = min(total_frames, max_frames) // process_interval
            estimated_time_seconds = frames_to_process * 0.2
            estimated_minutes = estimated_time_seconds / 60
            
            st.info(f"예상 처리 프레임: {frames_to_process:,}개 | 예상 처리 시간: {estimated_minutes:.1f}분")
            
            if st.button("비디오 처리 시작", type="primary", key="start_processing"):
                st.session_state['is_processing'] = True
                st.session_state['processing_complete'] = False
                st.session_state['auto_show_results'] = True  # 처리 완료 후 자동 표시
                st.session_state['process_params'] = {
                    'video_path': video_path,
                    'fps': fps,
                    'total_frames': total_frames,
                    'process_interval': process_interval,
                    'max_frames': max_frames,
                    'confidence': confidence
                }
                st.rerun()
        
        # 처리 실행
        if st.session_state.get('is_processing', False):
            params = st.session_state.get('process_params', {})
            results = self.batch_process_video(**params)
            
            if results:
                st.session_state['batch_results'] = results
                st.session_state['processing_complete'] = True
                st.session_state['is_processing'] = False
                # 처리 완료 후 자동으로 결과 표시
                st.success("처리 완료! 결과를 자동으로 표시합니다.")
                st.rerun()
            else:
                st.session_state['is_processing'] = False
                st.error("처리 중 오류가 발생했습니다.")
    
    def batch_process_video(self, video_path, fps, total_frames, process_interval, max_frames, confidence):
        """비디오를 배치로 처리하고 결과 반환"""
        from video import video_detector
        
        frames_to_process = list(range(0, min(total_frames, max_frames), process_interval))
        total_process_frames = len(frames_to_process)
        
        st.markdown("### 처리 진행상황")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        processed_results = []
        processing_times = []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("비디오 파일을 열 수 없습니다.")
            return None
        
        try:
            for i, frame_num in enumerate(frames_to_process):
                progress = (i + 1) / total_process_frames
                progress_bar.progress(progress)
                status_text.text(f"처리 중: {i+1}/{total_process_frames} 프레임 (프레임 #{frame_num})")
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                start_time = time.time()
                result_img, detections = video_detector.detect_objects_optimized(frame, confidence)
                processing_time = time.time() - start_time
                
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                frame_time = frame_num / fps
                
                processed_results.append({
                    'frame_number': frame_num,
                    'time': frame_time,
                    'image': result_rgb,
                    'original_image': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                    'processed_image_bgr': result_img,
                    'detections': detections,
                    'processing_time': processing_time,
                    'people_count': len(video_detector.detected_people),
                    'segmented_count': len(video_detector.segmented_people),
                    'mask_count': len(video_detector.mask_wearers)
                })
                
                processing_times.append(processing_time)
                
                if (i + 1) % 10 == 0:
                    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
                    remaining_frames = total_process_frames - (i + 1)
                    estimated_remaining = remaining_frames * avg_time
                    status_text.text(f"처리 중: {i+1}/{total_process_frames} 프레임 - 예상 남은 시간: {estimated_remaining:.1f}초")
        
        except Exception as e:
            st.error(f"처리 중 오류 발생: {e}")
            return None
        finally:
            cap.release()
        
        progress_bar.progress(1.0)
        status_text.text(f"처리 완료! 총 {len(processed_results)}개 프레임 처리됨")
        
        return processed_results
    
    # def create_processed_video(self, results, fps):
    #     """처리된 프레임들로부터 비디오 파일 생성"""
    #     import io
        
    #     if not results:
    #         return None
            
    #     try:
    #         temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    #         temp_video_path = temp_video.name
    #         temp_video.close()
            
    #         first_frame = results[0]['processed_image_bgr']
    #         height, width = first_frame.shape[:2]
            
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
            
    #         for result in results:
    #             if 'processed_image_bgr' in result:
    #                 out.write(result['processed_image_bgr'])
            
    #         out.release()
            
    #         with open(temp_video_path, 'rb') as f:
    #             video_bytes = f.read()
            
    #         os.unlink(temp_video_path)
            
    #         return video_bytes
            
    #     except Exception as e:
    #         st.error(f"비디오 생성 오류: {e}")
    #         return None
    def create_processed_video(self, results, fps):
        """처리된 프레임들로부터 H.264 코덱을 사용한 비디오 파일 생성"""
        import io
        
        if not results:
            return None
            
        try:
            temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_video_path = temp_video.name
            temp_video.close()
            
            first_frame = results[0]['processed_image_bgr']
            height, width = first_frame.shape[:2]
            
            # H.264 코덱 지정 (더 호환성이 높은 방법들을 순서대로 시도)
            codecs_to_try = [
                ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264 직접 지정
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264 (Apple 호환)
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4 Part 2 (fallback)
                ('XVID', cv2.VideoWriter_fourcc(*'XVID'))   # Xvid (최종 fallback)
            ]
            
            video_writer = None
            used_codec = None
            
            for codec_name, fourcc in codecs_to_try:
                try:
                    test_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                    
                    # 테스트 프레임 작성해보기
                    if test_writer.isOpened():
                        test_writer.write(first_frame)
                        test_writer.release()
                        
                        # 파일이 제대로 생성되었는지 확인
                        if os.path.getsize(temp_video_path) > 0:
                            video_writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
                            used_codec = codec_name
                            st.info(f"비디오 코덱: {codec_name} 사용")
                            break
                    else:
                        test_writer.release()
                        
                except Exception as e:
                    st.warning(f"{codec_name} 코덱 사용 실패: {e}")
                    continue
            
            if video_writer is None or not video_writer.isOpened():
                st.error("지원되는 비디오 코덱을 찾을 수 없습니다.")
                return None
            
            try:
                # 모든 프레임 작성
                frames_written = 0
                for result in results:
                    if 'processed_image_bgr' in result:
                        video_writer.write(result['processed_image_bgr'])
                        frames_written += 1
                
                video_writer.release()
                
                # 생성된 파일 크기 확인
                file_size = os.path.getsize(temp_video_path)
                if file_size == 0:
                    st.error("비디오 파일 생성에 실패했습니다. (파일 크기: 0)")
                    return None
                
                st.success(f"비디오 생성 완료: {frames_written}개 프레임, {file_size/1024/1024:.2f}MB, 코덱: {used_codec}")
                
                # 파일 읽기
                with open(temp_video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # 임시 파일 삭제
                os.unlink(temp_video_path)
                
                return video_bytes
                
            except Exception as e:
                st.error(f"비디오 작성 중 오류: {e}")
                video_writer.release()
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                return None
                
        except Exception as e:
            st.error(f"비디오 생성 오류: {e}")
            return None
    
    
    def display_batch_results(self, results, processing_times):
        """배치 처리 결과 표시"""
        if not results:
            st.warning("처리된 결과가 없습니다.")
            return
        
        st.markdown("### 처리 결과")
        
        # 새로 처리하기 버튼
        if st.button("새 비디오 처리하기", key="new_video_btn"):
            st.session_state['processing_complete'] = False
            st.session_state['batch_results'] = None
            st.session_state['auto_show_results'] = True
            st.rerun()
        
        # 통계 표시
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("처리된 프레임", len(results))
        with col2:
            avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
            st.metric("평균 처리시간", f"{avg_time:.3f}초")
        with col3:
            total_people = sum(r['people_count'] for r in results)
            st.metric("총 탐지된 사람", total_people)
        with col4:
            total_masks = sum(r['mask_count'] for r in results)
            st.metric("총 마스크 탐지", total_masks)
        
        # 결과 탐색 인터페이스
        st.markdown("### 결과 탐색")
        
        if len(results) > 1:
            selected_idx = st.slider(
                "프레임 선택",
                0, len(results) - 1, 0,
                format="프레임 %d",
                key="frame_selector"
            )
        else:
            selected_idx = 0
        
        if 0 <= selected_idx < len(results):
            result = results[selected_idx]
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"#### 프레임 #{result['frame_number']} (시간: {result['time']:.2f}초)")
                
                tab1, tab2 = st.tabs(["처리 결과", "원본"])
                
                with tab1:
                    st.image(result['image'], use_container_width=True, caption="처리된 이미지")
                
                with tab2:
                    if 'original_image' in result:
                        st.image(result['original_image'], use_container_width=True, caption="원본 이미지")
                
                st.markdown("**탐지 정보:**")
                st.write(f"- 처리 시간: {result['processing_time']:.3f}초")
                st.write(f"- 탐지된 사람: {result['people_count']}명")
                st.write(f"- 세그멘테이션: {result['segmented_count']}명")
                st.write(f"- 마스크 착용: {result['mask_count']}명")
            
            with col2:
                st.markdown("#### 프레임 분석")
                
                if not result['detections'].empty:
                    detection_summary = result['detections']['name'].value_counts()
                    for obj_type, count in detection_summary.items():
                        st.write(f"- {obj_type}: {count}개")
                else:
                    st.write("탐지된 객체 없음")
                
                if not result['detections'].empty and 'confidence' in result['detections'].columns:
                    st.markdown("**신뢰도 분포**")
                    fig = px.histogram(
                        result['detections'], 
                        x='confidence',
                        title="",
                        nbins=10
                    )
                    fig.update_layout(height=200, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key="confidence_histogram")
        
        # 전체 처리 통계 차트
        st.markdown("### 전체 분석 결과")
        
        if len(results) > 1:
            # 시간별 처리 성능 차트
            times_df = pd.DataFrame({
                '프레임': [r['frame_number'] for r in results],
                '처리시간': [r['processing_time'] for r in results],
                '탐지된사람': [r['people_count'] for r in results],
                '마스크착용': [r['mask_count'] for r in results]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 처리 시간 추이
                fig1 = px.line(times_df, x='프레임', y='처리시간', title="프레임별 처리 시간")
                st.plotly_chart(fig1, use_container_width=True, key="processing_time_chart")
            
            with col2:
                # 탐지 결과 추이
                fig2 = px.line(times_df, x='프레임', y=['탐지된사람', '마스크착용'], 
                              title="프레임별 탐지 결과")
                st.plotly_chart(fig2, use_container_width=True, key="detection_results_chart")
        
        # 결과 다운로드 옵션
        st.markdown("### 결과 저장")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV 다운로드 - 직접 다운로드 버튼
            summary_data = []
            for r in results:
                summary_data.append({
                    '프레임번호': r['frame_number'],
                    '시간': f"{r['time']:.2f}",
                    '처리시간': f"{r['processing_time']:.3f}",
                    '탐지된사람': r['people_count'],
                    '세그멘테이션': r['segmented_count'],
                    '마스크착용': r['mask_count']
                })
            
            df_summary = pd.DataFrame(summary_data)
            csv = df_summary.to_csv(index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="CSV 파일 다운로드",
                data=csv,
                file_name=f"video_analysis_results_{int(time.time())}.csv",
                mime="text/csv",
                key="results_csv_download"
            )
        
        with col2:
            # 비디오 다운로드
            if st.button("처리된 비디오 생성", key="create_video_btn"):
                with st.spinner("비디오 생성 중..."):
                    # FPS 계산
                    fps = 30.0
                    if len(results) > 1:
                        time_diff = results[1]['time'] - results[0]['time']
                        if time_diff > 0:
                            fps = min(1.0 / time_diff, 30.0)
                    
                    video_bytes = self.create_processed_video(results, fps)
                    
                    if video_bytes:
                        st.download_button(
                            label="MP4 파일 다운로드",
                            data=video_bytes,
                            file_name=f"processed_video_{int(time.time())}.mp4",
                            mime="video/mp4",
                            key="results_video_download"
                        )
                        st.success("비디오 파일이 준비되었습니다!")
                    else:
                        st.error("비디오 생성에 실패했습니다.")
            
    def render_video_player_controls(self, settings):
        """비디오 플레이어 컨트롤 UI 렌더링"""
        st.markdown("### 재생 컨트롤")
        
        col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
        # 재생/일시정지 버튼
        with col1:
            play_button_text = "일시정지" if st.session_state.is_playing else "재생"
            if st.button(play_button_text, type="primary", key="play_button"):
                if not st.session_state.is_playing:
                    st.session_state.is_playing = True
                    st.session_state.stop_video = False
                    if st.session_state.video_playing_thread is None or not st.session_state.video_playing_thread.is_alive():
                        st.session_state.video_playing_thread = threading.Thread(
                            target=self.video_playback_thread,
                            args=(st.session_state.video_path, settings['target_fps'], True, settings['confidence_threshold'])
                        )
                        st.session_state.video_playing_thread.daemon = True
                        st.session_state.video_playing_thread.start()
                else:
                    st.session_state.is_playing = False
                st.rerun()
        
        # 정지 버튼
        with col2:
            if st.button("정지", key="stop_button"):
                st.session_state.is_playing = False
                st.session_state.stop_video = True
                st.session_state.current_frame = 0
                st.rerun()
        
        # 재생 속도 선택
        with col3:
            speed_options = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
            speed_index = speed_options.index(st.session_state.playback_speed) if st.session_state.playback_speed in speed_options else 2
            new_speed = st.selectbox(
                "속도",
                speed_options,
                index=speed_index,
                format_func=lambda x: f"{x}배속",
                key="playback_speed_selector"
            )
            st.session_state.playback_speed = new_speed
        
        # 실시간 탐지 활성화 체크박스
        with col4:
            enable_detection = st.checkbox("실시간 탐지", value=True, key="enable_detection_checkbox")
        
        # 분석 표시 체크박스
        with col5:
            show_analytics = st.checkbox("분석", value=True, key="show_analytics_checkbox")
        
        # 프로그레스 바 (비디오 위치 조절)
        if st.session_state.total_frames > 0:
            progress = st.slider(
                "위치",
                0, st.session_state.total_frames - 1,
                st.session_state.current_frame,
                format="프레임 %d",
                key="video_progress_slider"
            )
            
            # 사용자가 슬라이더를 조작한 경우
            if progress != st.session_state.current_frame:
                st.session_state.current_frame = progress
                st.session_state.is_playing = False
                st.session_state.stop_video = True
        
        return enable_detection, show_analytics
    
    def render_video_display(self, enable_detection=True, confidence_threshold=0.5):
        """비디오 프레임 표시 영역"""
        if not st.session_state.video_path:
            return
        
        from video import video_detector
        
        # 현재 프레임 위치에서 프레임 가져오기
        frame = video_detector.get_frame_at_position(
            st.session_state.video_path,
            st.session_state.current_frame
        )
        
        if frame is not None:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown("### 비디오 프레임 분석")
                
                # 프레임 처리 시작
                start_time = time.time()
                if enable_detection:
                    result_img, detections = video_detector.detect_objects_optimized(frame, confidence_threshold)
                else:
                    result_img = frame
                    detections = pd.DataFrame()
                processing_time = time.time() - start_time
                
                # BGR에서 RGB로 변환하여 결과 표시
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # 프레임 정보를 포함한 캡션 생성
                current_time = st.session_state.current_frame / st.session_state.fps
                caption = f"프레임 {st.session_state.current_frame + 1} | 시간: {current_time:.2f}초 | 객체: {len(detections)} | 처리시간: {processing_time:.3f}초"
                
                st.image(result_rgb, caption=caption, use_container_width=True)
                
                # 재생 중일 때 자동 새로고침
                if st.session_state.is_playing:
                    time.sleep(0.1)
                    st.rerun()
            
            with col2:
                st.markdown("### 프레임 분석")
                
                # 실시간 통계 표시
                stages = ['사람 탐지', '세그멘테이션', '마스크 탐지']
                counts = [
                    len(video_detector.detected_people),
                    len(video_detector.segmented_people),
                    len(video_detector.mask_wearers)
                ]
                
                # 각 단계별 상태 표시
                for stage, count in zip(stages, counts):
                    color = "녹색" if count > 0 else "빨간색"
                    st.write(f"{color} {stage}: {count}")
                
                # 처리 시간 히스토리 관리
                if 'processing_times' not in st.session_state:
                    st.session_state.processing_times = []
                
                if enable_detection:
                    st.session_state.processing_times.append(processing_time)
                    # 최근 50개 기록만 유지
                    if len(st.session_state.processing_times) > 50:
                        st.session_state.processing_times.pop(0)
                
                # 처리 시간 추이 미니 차트
                if len(st.session_state.processing_times) > 5:
                    fig = px.line(
                        y=st.session_state.processing_times[-20:],
                        title="처리 시간 추이",
                        labels={'y': '시간 (초)', 'index': '프레임'}
                    )
                    fig.update_layout(height=200, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key="realtime_processing_chart")
                
                # 성능 메트릭 업데이트 (탐지가 활성화된 경우만)
                if enable_detection:
                    self.update_performance_metrics(processing_time, len(detections))
    
    def update_performance_metrics(self, processing_time, detection_count):
        """성능 메트릭 업데이트"""
        metrics = st.session_state.performance_metrics
        metrics['total_processed'] += 1
        
        # 평균 처리 시간 계산 (점진적 평균)
        if metrics['avg_processing_time'] == 0:
            metrics['avg_processing_time'] = processing_time
        else:
            metrics['avg_processing_time'] = (metrics['avg_processing_time'] + processing_time) / 2
    
    def render_analytics_dashboard(self):
        """분석 대시보드 렌더링"""
        st.markdown("## 분석 대시보드")
        
        # 배치 처리 결과와 실시간 처리 결과 통합 확인
        has_batch_data = st.session_state.get('batch_results') is not None and len(st.session_state.get('batch_results', [])) > 0
        has_realtime_data = st.session_state.get('processing_times') is not None and len(st.session_state.get('processing_times', [])) > 1
        
        if not has_batch_data and not has_realtime_data:
            st.info("처리 데이터가 없습니다. 이미지나 비디오를 분석해주세요.")
            return
        
        # 데이터 소스 선택 탭
        if has_batch_data and has_realtime_data:
            data_tab1, data_tab2 = st.tabs(["배치 처리 결과", "실시간 처리 결과"])
            
            with data_tab1:
                self.render_batch_analytics()
            
            with data_tab2:
                self.render_realtime_analytics()
        elif has_batch_data:
            st.markdown("### 배치 처리 분석 결과")
            self.render_batch_analytics()
        else:
            st.markdown("### 실시간 처리 분석 결과")
            self.render_realtime_analytics()
    
    def render_batch_analytics(self):
        """배치 처리 결과 분석 렌더링"""
        batch_results = st.session_state.get('batch_results', [])
        
        if not batch_results:
            st.warning("배치 처리 결과가 없습니다.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 배치 처리 성능 차트
            batch_df = pd.DataFrame({
                '프레임번호': [r['frame_number'] for r in batch_results],
                '처리시간': [r['processing_time'] for r in batch_results],
                '탐지된사람': [r['people_count'] for r in batch_results],
                '세그멘테이션': [r['segmented_count'] for r in batch_results],
                '마스크착용': [r['mask_count'] for r in batch_results],
                '시간': [r['time'] for r in batch_results]
            })
            
            # 처리 성능 차트
            fig1 = px.line(batch_df, x='시간', y='처리시간', 
                          title="시간별 처리 성능 (배치 처리)")
            fig1.update_layout(height=300)
            st.plotly_chart(fig1, use_container_width=True, key="batch_performance_chart")
            
            # 탐지 결과 추이 차트
            fig2 = px.line(batch_df, x='시간', y=['탐지된사람', '세그멘테이션', '마스크착용'], 
                          title="시간별 탐지 결과 추이")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True, key="batch_detection_trend")
            
        with col2:
            # 배치 처리 통계
            st.markdown("### 배치 처리 통계")
            
            processing_times = [r['processing_time'] for r in batch_results]
            total_people = sum(r['people_count'] for r in batch_results)
            total_segments = sum(r['segmented_count'] for r in batch_results)
            total_masks = sum(r['mask_count'] for r in batch_results)
            
            st.metric("처리된 프레임", len(batch_results))
            st.metric("총 처리 시간", f"{sum(processing_times):.2f}초")
            st.metric("평균 처리 시간", f"{np.mean(processing_times):.3f}초")
            st.metric("최대 처리 시간", f"{max(processing_times):.3f}초")
            st.metric("최소 처리 시간", f"{min(processing_times):.3f}초")
            
            st.markdown("### 탐지 통계")
            st.metric("총 사람 탐지", total_people)
            st.metric("총 세그멘테이션", total_segments)
            st.metric("총 마스크 탐지", total_masks)
            
            # 탐지 성공률 계산
            if len(batch_results) > 0:
                people_success_rate = (sum(1 for r in batch_results if r['people_count'] > 0) / len(batch_results)) * 100
                mask_success_rate = (sum(1 for r in batch_results if r['mask_count'] > 0) / len(batch_results)) * 100
                
                st.markdown("### 성공률")
                st.metric("사람 탐지 성공률", f"{people_success_rate:.1f}%")
                st.metric("마스크 탐지 성공률", f"{mask_success_rate:.1f}%")
    
    def render_realtime_analytics(self):
        """실시간 처리 결과 분석 렌더링"""
        processing_times = st.session_state.get('processing_times', [])
        
        if len(processing_times) < 2:
            st.warning("실시간 처리 데이터가 충분하지 않습니다.")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 실시간 처리 히스토리 차트
            df_times = pd.DataFrame({
                '프레임': range(len(processing_times)),
                '처리시간': processing_times
            })
            fig = px.line(df_times, x='프레임', y='처리시간', 
                         title="실시간 처리 성능")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True, key="realtime_performance_chart")
        
        with col2:
            # 실시간 처리 통계
            st.markdown("### 실시간 처리 통계")
            metrics = st.session_state.performance_metrics
            
            st.metric("총 처리됨", metrics['total_processed'])
            st.metric("평균 처리시간", f"{metrics['avg_processing_time']:.3f}초")
            
            # 현재 세션의 처리 시간 통계
            avg_time = np.mean(processing_times)
            max_time = max(processing_times)
            min_time = min(processing_times)
            
            st.metric("현재 세션 평균", f"{avg_time:.3f}초")
            st.metric("최대 처리 시간", f"{max_time:.3f}초")
            st.metric("최소 처리 시간", f"{min_time:.3f}초")
            
            # 처리 시간 분포 히스토그램
            if len(processing_times) > 10:
                st.markdown("### 처리 시간 분포")
                fig_hist = px.histogram(
                    x=processing_times,
                    nbins=20,
                    title="처리 시간 분포"
                )
                fig_hist.update_layout(height=250, showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True, key="processing_time_histogram")
    
    def run(self):
        """메인 UI 실행 함수"""
        # 메인 헤더 렌더링
        self.render_header()
        
        # 사이드바에서 설정값들 가져오기
        settings = self.render_advanced_sidebar()
        
        # 처리 파이프라인 시각화 렌더링
        self.render_processing_pipeline_visual()
        
        # 메인 탭 구성 (이미지 분석, 비디오 분석, 실시간 분석)
        tab1, tab2, tab3 = st.tabs(["이미지 분석", "비디오 분석", "실시간 분석"])
        
        with tab1:
            self.render_image_detection_mode(settings)
        
        with tab2:
            self.render_video_analysis_mode(settings)
        
        with tab3:
            self.render_analytics_dashboard()

# UI 실행 함수
def run_video_ui():
    """고급 UI 실행 진입점"""
    try:
        ui = PremiumUI()
        ui.run_standalone()
    except Exception as e:
        st.error(f"애플리케이션 오류: {e}")
        st.info("페이지를 새로고침하거나 설정을 확인해주세요.")

if __name__ == "__main__":
    run_video_ui()