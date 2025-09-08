import streamlit as st
import time
import os
from datetime import datetime
from PIL import Image
from pathlib import Path
from camera import camera_controller
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import tempfile

def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'camera_on': False,
        'auto_scan': False,
        'last_scan_time': 0,
        'select_all_files': False,
        # 객체 탐지 관련 상태
        'detection_results': None,
        'processed_images': {},
        'processing_times': [],
        'performance_metrics': {
            'total_processed': 0,
            'avg_processing_time': 0
        },
        # 실시간 탐지 상태
        'realtime_enabled': False,
        'capture_frame': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_camera_control_tab():
    """카메라 제어 탭 렌더링"""
    col1, col2 = st.columns([1, 1])
    
    system_info = camera_controller.get_system_info()
    
    with col1:
        st.info(f"**운영체제**: {system_info['os']}")
    with col2:
        camera_running, running_processes = camera_controller.check_camera_status()
        if camera_running:
            st.success(f"🟢 카메라 앱 실행 중 ({len(running_processes)}개)")
        else:
            st.error("🔴 카메라 앱 없음")

    # 컨트롤 패널
    st.markdown("### 🎮 카메라 제어")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("🎥 카메라 실행", use_container_width=True, type="primary"):
            success, message = camera_controller.start_camera()
            if success:
                st.success(message)
                st.session_state.camera_on = True
            else:
                st.error(message)
            time.sleep(0.5)
            st.rerun()

    with col2:
        if st.button("🔴 카메라 종료", use_container_width=True, type="secondary"):
            success, message = camera_controller.stop_camera()
            if success:
                st.info(message)
            else:
                st.error(message)
            st.session_state.camera_on = False
            time.sleep(0.5)
            st.rerun()

    with col3:
        if st.button("🔄 상태 새로고침", use_container_width=True):
            st.rerun()

    # 실행 중인 카메라 앱 정보
    if running_processes:
        st.markdown("### 📊 실행 중인 카메라 앱")
        for i, proc in enumerate(running_processes, 1):
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{i}.** {proc['name']}")
            with col2:
                st.write(f"PID: {proc['pid']}")
            with col3:
                st.write(f"메모리: {proc['memory']}")

def render_file_management_tab():
    """파일 관리 탭 렌더링"""
    st.markdown("### 📁 카메라 파일 관리")
    
    # 파일 스캔 설정
    col1, col2, col3 = st.columns(3)
    with col1:
        scan_hours = st.selectbox(
            "스캔 범위",
            [1, 6, 12, 24, 48, 72],
            index=3,
            format_func=lambda x: f"최근 {x}시간"
        )
    
    with col2:
        auto_scan = st.checkbox("자동 스캔", value=st.session_state.auto_scan)
        st.session_state.auto_scan = auto_scan
    
    with col3:
        if st.button("🔍 파일 스캔", type="primary"):
            st.session_state.last_scan_time = time.time()
            st.rerun()
    
    # 기본 카메라 폴더 가져오기
    default_folders = camera_controller.get_default_camera_folders()
    
    if not default_folders:
        st.warning("기본 카메라 폴더를 찾을 수 없습니다.")
    else:
        st.info(f"스캔 대상 폴더 ({len(default_folders)}개): " + 
               ", ".join([f.name for f in default_folders[:3]]) + 
               ("..." if len(default_folders) > 3 else ""))
    
    # 파일 스캔 및 표시
    if auto_scan or st.session_state.last_scan_time > 0:
        with st.spinner("파일을 스캔하는 중..."):
            found_files = camera_controller.scan_camera_files(
                folders=default_folders, 
                limit_hours=scan_hours
            )
        
        if found_files:
            st.success(f"📸 {len(found_files)}개의 파일을 찾았습니다!")
            render_file_list(found_files)
        else:
            st.info(f"최근 {scan_hours}시간 내 카메라 파일이 없습니다.")
    
    # 자동 새로고침
    if auto_scan:
        st.info("🔄 자동 스캔이 활성화되어 있습니다. (30초마다 새로고침)")
        time.sleep(30)
        st.rerun()

def render_file_list(found_files):
    """파일 목록 렌더링"""
    # 파일 필터링
    col1, col2 = st.columns(2)
    with col1:
        file_type_filter = st.selectbox(
            "파일 타입 필터",
            ["모두", "이미지만", "비디오만"]
        )
    
    with col2:
        show_previews = st.checkbox("미리보기 표시", value=True)
    
    # 필터 적용
    if file_type_filter == "이미지만":
        filtered_files = [f for f in found_files if f['type'] == 'image']
    elif file_type_filter == "비디오만":
        filtered_files = [f for f in found_files if f['type'] == 'video']
    else:
        filtered_files = found_files
    
    if not filtered_files:
        st.info("선택한 필터에 해당하는 파일이 없습니다.")
        return
    
    # 파일 선택 및 표시
    st.markdown("### 📋 파일 목록")
    selected_files = []
    
    # 전체 선택/해제 버튼
    col1, col2 = st.columns([1, 3])
    with col1:
        select_all = st.button("✅ 전체 선택")
        if select_all:
            st.session_state.select_all_files = True
    
    # 파일 목록 표시
    for i, file_info in enumerate(filtered_files[:20]):
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            is_selected = st.checkbox(
                "선택",
                key=f"file_{i}",
                value=getattr(st.session_state, 'select_all_files', False)
            )
            if is_selected:
                selected_files.append(file_info)
        
        with col2:
            file_time = datetime.fromtimestamp(file_info['modified'])
            st.write(f"**{file_info['name']}**")
            st.caption(f"📁 {file_info['folder']} | 📅 {file_time.strftime('%Y-%m-%d %H:%M')} | 📄 {camera_controller.format_file_size(file_info['size'])}")
        
        with col3:
            if show_previews and file_info['type'] == 'image':
                try:
                    img = Image.open(file_info['path'])
                    img.thumbnail((100, 100))
                    st.image(img, caption="미리보기")
                except:
                    st.write("🖼️ 이미지")
            elif file_info['type'] == 'video':
                st.write("🎬 비디오")
    
    # 선택된 파일이 있을 때 작업 옵션
    if selected_files:
        render_file_actions(selected_files)
    
    # 20개 이상일 때 알림
    if len(filtered_files) > 20:
        st.info(f"📊 총 {len(filtered_files)}개 파일 중 최신 20개만 표시됩니다.")

def render_file_actions(selected_files):
    """선택된 파일에 대한 작업 옵션"""
    st.markdown(f"### 🎯 선택된 파일 ({len(selected_files)}개)")
    
    # 저장 경로 설정
    save_path = st.text_input(
        "저장 경로",
        value=camera_controller.save_directory,
        help="파일을 저장할 폴더 경로"
    )
    camera_controller.save_directory = save_path
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        organize_by_date = st.checkbox("날짜별 폴더 정리")
    
    with col2:
        if st.button("📁 폴더 생성"):
            success, message = camera_controller.create_save_directory(save_path)
            if success:
                st.success(message)
            else:
                st.error(message)
    
    with col3:
        if st.button("💾 파일 복사", type="primary"):
            success, copied_files, errors = camera_controller.copy_files_to_destination(
                selected_files, save_path, organize_by_date
            )
            
            if success:
                st.success(f"✅ {len(copied_files)}개 파일이 복사되었습니다!")
                if copied_files:
                    with st.expander("복사된 파일 목록"):
                        for filename in copied_files:
                            st.write(f"• {filename}")
                
                if errors:
                    st.warning(f"⚠️ {len(errors)}개 파일에서 오류 발생:")
                    for error in errors:
                        st.write(f"• {error}")
            else:
                st.error("파일 복사에 실패했습니다.")
    
    with col4:
        if st.button("📦 ZIP 다운로드"):
            success, zip_path = camera_controller.create_zip_archive(selected_files)
            if success:
                with open(zip_path, "rb") as zip_file:
                    st.download_button(
                        label="📥 ZIP 파일 다운로드",
                        data=zip_file.read(),
                        file_name=f"camera_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )
                os.unlink(zip_path)
            else:
                st.error(f"ZIP 생성 실패: {zip_path}")

def render_unified_detection_tab():
    """통합된 카메라 탐지 탭 렌더링"""
    st.markdown("### 🔬 실시간 카메라 객체 탐지")
    
    # 시스템 정보 표시
    system_info = camera_controller.get_system_info()
    col1, col2, col3 = st.columns(3)
    with col1:
        device_status = "GPU 가속" if system_info['cuda_available'] else "CPU 모드"
        device_name = system_info['device_name']
        st.info(f"**처리 장치**: {device_status} ({device_name})")
    
    with col2:
        model_status = "로드됨" if camera_controller.model is not None else "대기 중"
        st.info(f"**모델 상태**: {model_status}")
    
    with col3:
        realtime_status = "활성화" if camera_controller.is_realtime_active() else "비활성화"
        status_color = "success" if camera_controller.is_realtime_active() else "error"
        getattr(st, status_color)(f"**실시간 탐지**: {realtime_status}")
    
    # 처리 파이프라인 시각화
    render_detection_pipeline_visual()
    
    # 카메라 모드 선택
    detection_mode = st.radio(
        "카메라 모드 선택",
        ["실시간 카메라", "즉시 촬영", "파일 분석"],
        horizontal=True
    )
    
    st.markdown("---")
    
    if detection_mode == "실시간 카메라":
        render_realtime_camera_mode()
    elif detection_mode == "즉시 촬영":
        render_instant_photo_mode()
    else:  # 파일 분석
        render_file_analysis_mode()

def render_realtime_camera_mode():
    """실시간 카메라 모드 - OpenCV 윈도우 사용"""
    st.markdown("#### 🎥 실시간 카메라 탐지 (OpenCV 윈도우)")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**설정**")
        
        # 사용 가능한 카메라 목록
        available_cameras = camera_controller.get_available_cameras()
        
        if not available_cameras:
            st.error("사용 가능한 카메라를 찾을 수 없습니다.")
            st.info("카메라가 연결되어 있는지 확인하고 다른 앱에서 사용 중이 아닌지 확인해주세요.")
            return
        
        # 카메라 선택
        camera_options = [f"카메라 {cam['index']} ({cam['resolution']})" for cam in available_cameras]
        selected_camera_idx = st.selectbox(
            "카메라 선택",
            range(len(camera_options)),
            format_func=lambda x: camera_options[x],
            help="사용할 카메라 선택"
        )
        selected_camera = available_cameras[selected_camera_idx]['index']
        
        # 신뢰도 설정
        confidence = st.slider(
            "신뢰도 임계값",
            0.3, 0.8, 0.5, 0.1,
            help="탐지 신뢰도 설정"
        )
        
        # 모델 선택
        model_choice = st.selectbox(
            "모델 선택",
            ["yolov8n-seg.pt", "mask_best"],
            help="사용할 탐지 모델"
        )
        
        # 탐지 간격 설정
        detection_interval = st.slider(
            "탐지 간격 (프레임)",
            1, 10, 3,
            help="N 프레임마다 객체 탐지 수행"
        )
        
        # 모델 로드
        if model_choice != camera_controller.selected_model:
            with st.spinner("모델 로딩 중..."):
                camera_controller.load_model(model_choice)
        
        st.markdown("**제어**")
        
        # OpenCV 윈도우 실시간 탐지 시작
        if st.button("🎥 OpenCV 실시간 탐지 시작", type="primary", use_container_width=True):
            with st.spinner("실시간 탐지 창을 여는 중..."):
                success, message = start_opencv_realtime_detection(
                    selected_camera, confidence, detection_interval
                )
            if success:
                st.success(message)
                st.info("OpenCV 창에서 'q' 키를 누르면 종료됩니다.")
            else:
                st.error(message)
        
        # 단일 프레임 캡처 및 분석
        if st.button("📸 단일 프레임 캡처 & 분석", use_container_width=True):
            success, result_data = capture_and_analyze_single_frame(selected_camera, confidence)
            if success:
                st.session_state['single_frame_result'] = result_data
                st.success("프레임이 캡처되고 분석되었습니다!")
                st.rerun()
            else:
                st.error("프레임 캡처에 실패했습니다.")
    
    with col1:
        st.markdown("**실시간 카메라 정보**")
        
        # 시스템 정보
        system_info = camera_controller.get_system_info()
        col_a, col_b = st.columns(2)
        with col_a:
            device_status = "GPU 가속" if system_info['cuda_available'] else "CPU 모드"
            st.info(f"처리 장치: {device_status}")
        with col_b:
            model_status = "로드됨" if camera_controller.model is not None else "대기 중"
            st.info(f"모델 상태: {model_status}")
        
        # OpenCV 창 사용법 안내
        st.markdown("**사용법:**")
        st.write("1. 설정을 조정한 후 '실시간 탐지 시작' 버튼을 클릭")
        st.write("2. OpenCV 창이 열리면서 실시간 객체 탐지 시작")
        st.write("3. 창에서 'q' 키를 누르면 종료")
        st.write("4. 'ESC' 키로 일시정지/재개 가능")
        
        # 단일 프레임 결과 표시
        if 'single_frame_result' in st.session_state:
            st.markdown("**최근 캡처 분석 결과:**")
            result = st.session_state['single_frame_result']
            
            # 이미지 표시
            if result['image'] is not None:
                st.image(result['image'], caption=f"분석 결과 - 처리시간: {result['processing_time']:.3f}초", use_container_width=True)
            
            # 통계 표시
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("탐지된 사람", result['people_count'])
            with col_info2:
                st.metric("세그멘테이션", result['segmented_count'])
            with col_info3:
                st.metric("마스크 착용", result['mask_count'])
        
        # 카메라 테스트
        if st.button("🎹 카메라 연결 테스트"):
            available, message = camera_controller.check_camera_availability(selected_camera)
            if available:
                st.success(f"카메라 {selected_camera}를 사용할 수 있습니다!")
            else:
                st.error(f"카메라 {selected_camera}: {message}")

def start_opencv_realtime_detection(camera_index, confidence_threshold, detection_interval):
    """OpenCV 윈도우에서 실시간 객체 탐지 실행"""
    import threading
    
    def detection_worker():
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return False, "카메라를 열 수 없습니다"
        
        # 카메라 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # 윈도우 설정
        window_name = "실시간 객체 탐지 - 'q':종료, 'ESC':일시정지"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        frame_count = 0
        last_detection_result = None
        paused = False
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # 탐지 수행 (지정된 간격마다)
                    if frame_count % detection_interval == 0:
                        result_img, detections = camera_controller.detect_objects_optimized(
                            frame, confidence_threshold
                        )
                        last_detection_result = result_img
                    else:
                        # 이전 탐지 결과 사용
                        result_img = last_detection_result if last_detection_result is not None else frame
                    
                    # FPS 및 탐지 정보 오버레이
                    info_text = [
                        f"FPS: {cap.get(cv2.CAP_PROP_FPS):.1f}",
                        f"Frame: {frame_count}",
                        f"People: {len(camera_controller.detected_people)}",
                        f"Segmented: {len(camera_controller.segmented_people)}",
                        f"Masks: {len(camera_controller.mask_wearers)}",
                        f"Confidence: {confidence_threshold}",
                        "'q': 종료, 'ESC': 일시정지/재개"
                    ]
                    
                    for i, text in enumerate(info_text):
                        y_pos = 30 + (i * 25)
                        cv2.putText(result_img, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(result_img, text, (10, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    cv2.imshow(window_name, result_img)
                    frame_count += 1
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 'q' 키로 종료
                    break
                elif key == 27:  # ESC 키로 일시정지/재개
                    paused = not paused
                    if paused:
                        cv2.putText(result_img, "PAUSED - Press ESC to resume", 
                                  (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow(window_name, result_img)
        
        except Exception as e:
            print(f"실시간 탐지 중 오류: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    # 별도 스레드에서 실행
    thread = threading.Thread(target=detection_worker, daemon=True)
    thread.start()
    
    return True, "실시간 탐지가 시작되었습니다. OpenCV 창을 확인해주세요."

def capture_and_analyze_single_frame(camera_index, confidence_threshold):
    """단일 프레임 캡처 및 분석"""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        return False, None
    
    try:
        # 프레임 캡처
        ret, frame = cap.read()
        if not ret:
            return False, None
        
        # 객체 탐지 수행
        start_time = time.time()
        result_img, detections = camera_controller.detect_objects_optimized(frame, confidence_threshold)
        processing_time = time.time() - start_time
        
        # RGB로 변환 (Streamlit 표시용)
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        # 결과 데이터 구성
        result_data = {
            'image': result_rgb,
            'processing_time': processing_time,
            'people_count': len(camera_controller.detected_people),
            'segmented_count': len(camera_controller.segmented_people),
            'mask_count': len(camera_controller.mask_wearers),
            'detections': detections
        }
        
        # 성능 메트릭 업데이트
        update_performance_metrics(processing_time, len(detections) if isinstance(detections, pd.DataFrame) else 0)
        
        return True, result_data
        
    except Exception as e:
        print(f"프레임 캡처 오류: {e}")
        return False, None
    finally:
        cap.release()

def render_instant_photo_mode():
    """즉시 촬영 모드"""
    st.markdown("#### 📷 즉시 촬영 및 분석")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**촬영 설정**")
        
        confidence = st.slider(
            "신뢰도 임계값",
            0.3, 0.8, 0.5, 0.1,
            help="탐지 신뢰도 설정",
            key="instant_confidence"
        )
        
        model_choice = st.selectbox(
            "모델 선택",
            ["yolov8n-seg.pt", "mask_best"],
            help="사용할 탐지 모델",
            key="instant_model"
        )
        
        # 모델 로드
        if model_choice != camera_controller.selected_model:
            with st.spinner("모델 로딩 중..."):
                camera_controller.load_model(model_choice)
        
        auto_analyze = st.checkbox("자동 분석", value=True, help="촬영 즉시 자동으로 분석")
    
    with col1:
        st.markdown("**즉시 촬영**")
        
        # Streamlit의 camera_input 사용
        camera_photo = st.camera_input("카메라로 사진 촬영")
        
        if camera_photo is not None:
            if auto_analyze:
                # 자동 분석 수행
                with st.spinner("촬영된 이미지 분석 중..."):
                    start_time = time.time()
                    result_img, detections, message = camera_controller.process_streamlit_camera_input(
                        camera_photo, confidence
                    )
                    processing_time = time.time() - start_time
                
                if result_img is not None:
                    # 결과 표시
                    tab1, tab2 = st.tabs(["분석 결과", "원본"])
                    
                    with tab1:
                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True, 
                                caption=f"객체 탐지 결과 - 처리시간: {processing_time:.3f}초")
                    
                    with tab2:
                        st.image(camera_photo, use_container_width=True, caption="촬영된 원본")
                    
                    # 탐지 정보 표시
                    col_result1, col_result2, col_result3 = st.columns(3)
                    with col_result1:
                        st.metric("탐지된 사람", len(camera_controller.detected_people))
                    with col_result2:
                        st.metric("세그멘테이션", len(camera_controller.segmented_people))
                    with col_result3:
                        st.metric("마스크 착용", len(camera_controller.mask_wearers))
                    
                    # 상세 분석 정보
                    if isinstance(detections, pd.DataFrame) and not detections.empty:
                        with st.expander("상세 분석 결과"):
                            st.dataframe(detections, use_container_width=True)
                    
                    # 성능 메트릭 업데이트
                    update_performance_metrics(processing_time, len(detections) if isinstance(detections, pd.DataFrame) else 0)
                    
                    st.success(f"분석 완료! 처리 시간: {processing_time:.3f}초")
                else:
                    st.error(f"분석 실패: {message}")
            else:
                # 수동 분석
                st.image(camera_photo, use_container_width=True, caption="촬영된 사진")
                
                if st.button("🔍 이미지 분석하기", type="primary"):
                    with st.spinner("이미지 분석 중..."):
                        start_time = time.time()
                        result_img, detections, message = camera_controller.process_streamlit_camera_input(
                            camera_photo, confidence
                        )
                        processing_time = time.time() - start_time
                    
                    if result_img is not None:
                        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                        st.image(result_rgb, use_container_width=True, caption="분석 결과")
                        st.success(f"분석 완료! 처리 시간: {processing_time:.3f}초")
                    else:
                        st.error(f"분석 실패: {message}")

def render_file_analysis_mode():
    """파일 분석 모드"""
    st.markdown("#### 📁 저장된 파일 분석")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.markdown("**분석 설정**")
        
        confidence = st.slider(
            "신뢰도 임계값",
            0.3, 0.8, 0.5, 0.1,
            help="탐지 신뢰도 설정",
            key="file_confidence"
        )
        
        model_choice = st.selectbox(
            "모델 선택",
            ["yolov8n-seg.pt", "mask_best"],
            help="사용할 탐지 모델",
            key="file_model"
        )
        
        show_original = st.checkbox("원본 이미지 비교", value=True, key="file_show_original")
    
    with col1:
        # 파일 업로드
        uploaded_file = st.file_uploader(
            "분석할 이미지/비디오 업로드",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'mp4', 'avi', 'mov', 'mkv'],
            help="카메라로 촬영한 이미지나 비디오를 업로드하세요"
        )
        
        if uploaded_file is not None:
            # 파일 타입에 따라 처리
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
                process_uploaded_image(uploaded_file, confidence, model_choice, show_original)
            elif file_extension in ['mp4', 'avi', 'mov', 'mkv']:
                process_uploaded_video(uploaded_file, confidence)

def render_detection_pipeline_visual():
    """객체 탐지 파이프라인 시각화"""
    st.markdown("#### 🔄 처리 파이프라인")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        people_count = len(camera_controller.detected_people)
        status = "🟢" if people_count > 0 else "🔴"
        st.markdown(f"""
        <div style="border-left: 4px solid #667eea; padding: 1rem; background: #f8f9fa; border-radius: 0 8px 8px 0;">
            <h5>{status} 1단계: 사람 탐지</h5>
            <p>L자형 경계상자</p>
            <strong>{people_count}명 탐지됨</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        seg_count = len(camera_controller.segmented_people)
        status = "🟢" if seg_count > 0 else "🔴"
        st.markdown(f"""
        <div style="border-left: 4px solid #667eea; padding: 1rem; background: #f8f9fa; border-radius: 0 8px 8px 0;">
            <h5>{status} 2단계: 세그멘테이션</h5>
            <p>픽셀 단위 분할 마스크</p>
            <strong>{seg_count}명 분할됨</strong>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mask_count = len(camera_controller.mask_wearers)
        status = "🟢" if mask_count > 0 else "🔴"
        st.markdown(f"""
        <div style="border-left: 4px solid #667eea; padding: 1rem; background: #f8f9fa; border-radius: 0 8px 8px 0;">
            <h5>{status} 3단계: 마스크 탐지</h5>
            <p>다이아몬드 커서 표시</p>
            <strong>{mask_count}개 마스크 탐지됨</strong>
        </div>
        """, unsafe_allow_html=True)

def process_uploaded_image(uploaded_file, confidence, model_choice, show_original):
    """업로드된 이미지 처리"""
    # 모델 설정
    if model_choice != camera_controller.selected_model:
        with st.spinner("모델 로딩 중..."):
            camera_controller.load_model(model_choice)
    
    # 이미지 로드
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # OpenCV 형식으로 변환 (RGB -> BGR)
    if len(img_array.shape) == 3:
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("**처리 결과**")
        
        # 처리 시작
        start_time = time.time()
        
        with st.spinner("순차 처리 파이프라인 실행 중..."):
            result_img, detections = camera_controller.detect_objects_optimized(img_bgr, confidence)
        
        processing_time = time.time() - start_time
        
        # 결과 표시
        if show_original:
            tab1, tab2 = st.tabs(["처리 결과", "원본"])
            with tab1:
                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True, caption="객체 탐지 결과")
            with tab2:
                st.image(image, use_container_width=True, caption="원본 이미지")
        else:
            result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, use_container_width=True, caption="객체 탐지 결과")
        
        st.success(f"처리 완료! 처리 시간: {processing_time:.3f}초")
        
        # 성능 메트릭 업데이트
        update_performance_metrics(processing_time, len(detections))
    
    with col2:
        st.markdown("**분석 보고서**")
        
        # 단계별 결과
        stages_data = {
            '단계': ['사람 탐지', '세그멘테이션', '마스크 탐지'],
            '개수': [
                len(camera_controller.detected_people),
                len(camera_controller.segmented_people),
                len(camera_controller.mask_wearers)
            ],
            '상태': ['✅' if count > 0 else '❌' for count in [
                len(camera_controller.detected_people),
                len(camera_controller.segmented_people),
                len(camera_controller.mask_wearers)
            ]]
        }
        
        df = pd.DataFrame(stages_data)
        st.dataframe(df, use_container_width=True)
        
        # 상세 정보
        if len(detections) > 0:
            st.markdown("**탐지 상세**")
            detection_summary = detections['name'].value_counts()
            for obj_type, count in detection_summary.items():
                st.write(f"• {obj_type}: {count}개")

def process_uploaded_video(uploaded_video, confidence):
    """업로드된 비디오 처리"""
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name
    
    try:
        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("비디오를 열 수 없습니다.")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 프레임", f"{total_frames:,}")
        with col2:
            st.metric("FPS", f"{fps:.1f}")
        with col3:
            st.metric("길이", f"{duration:.1f}초")
        
        # 첫 번째 프레임 처리
        ret, frame = cap.read()
        if ret:
            start_time = time.time()
            with st.spinner("비디오 첫 프레임 분석 중..."):
                result_img, detections = camera_controller.detect_objects_optimized(frame, confidence)
            processing_time = time.time() - start_time
            
            # 결과 표시
            col1, col2 = st.columns([2, 1])
            
            with col1:
                tab1, tab2 = st.tabs(["분석 결과", "원본 프레임"])
                with tab1:
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    st.image(result_rgb, use_container_width=True, caption="첫 프레임 분석 결과")
                with tab2:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    st.image(frame_rgb, use_container_width=True, caption="원본 첫 프레임")
            
            with col2:
                st.markdown("**분석 보고서**")
                
                # 처리 시간
                st.metric("처리 시간", f"{processing_time:.3f}초")
                
                # 단계별 결과
                stages_data = {
                    '단계': ['사람 탐지', '세그멘테이션', '마스크 탐지'],
                    '개수': [
                        len(camera_controller.detected_people),
                        len(camera_controller.segmented_people),
                        len(camera_controller.mask_wearers)
                    ]
                }
                
                df = pd.DataFrame(stages_data)
                st.dataframe(df, use_container_width=True)
            
            # 성능 메트릭 업데이트
            update_performance_metrics(processing_time, len(detections))
        else:
            st.error("비디오 첫 프레임을 읽을 수 없습니다.")
        
        cap.release()
        
    except Exception as e:
        st.error(f"비디오 처리 오류: {e}")
    finally:
        # 임시 파일 정리
        if os.path.exists(video_path):
            os.unlink(video_path)

def render_detection_analytics():
    """탐지 분석 결과 렌더링"""
    st.markdown("#### 📊 분석 통계")
    
    processing_times = st.session_state.get('processing_times', [])
    
    if not processing_times:
        st.info("아직 분석한 파일이 없습니다. 이미지나 비디오를 분석해보세요.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 처리 시간 추이 차트
        if len(processing_times) > 1:
            df_times = pd.DataFrame({
                '순서': range(1, len(processing_times) + 1),
                '처리시간': processing_times
            })
            
            fig = px.line(df_times, x='순서', y='처리시간', 
                         title="파일별 처리 시간 추이")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # 처리 시간 분포 히스토그램
            fig2 = px.histogram(x=processing_times, nbins=10, title="처리 시간 분포")
            fig2.update_layout(height=300)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("차트를 표시하려면 2개 이상의 파일을 분석해주세요.")
    
    with col2:
        # 통계 요약
        st.markdown("**통계 요약**")
        metrics = st.session_state.performance_metrics
        
        st.metric("총 처리된 파일", metrics['total_processed'])
        st.metric("평균 처리 시간", f"{metrics['avg_processing_time']:.3f}초")
        
        if processing_times:
            st.metric("최근 세션 평균", f"{sum(processing_times)/len(processing_times):.3f}초")
            st.metric("최대 처리 시간", f"{max(processing_times):.3f}초")
            st.metric("최소 처리 시간", f"{min(processing_times):.3f}초")

def update_performance_metrics(processing_time, detection_count):
    """성능 메트릭 업데이트"""
    # 처리 시간 기록
    if 'processing_times' not in st.session_state:
        st.session_state.processing_times = []
    
    st.session_state.processing_times.append(processing_time)
    
    # 최근 50개 기록만 유지
    if len(st.session_state.processing_times) > 50:
        st.session_state.processing_times.pop(0)
    
    # 성능 메트릭 업데이트
    metrics = st.session_state.performance_metrics
    metrics['total_processed'] += 1
    
    # 점진적 평균 계산
    if metrics['avg_processing_time'] == 0:
        metrics['avg_processing_time'] = processing_time
    else:
        metrics['avg_processing_time'] = (metrics['avg_processing_time'] + processing_time) / 2

def render_settings_tab():
    """설정 탭 렌더링"""
    st.markdown("### ⚙️ 설정")
    
    # 기본 저장 경로 설정
    st.markdown("#### 📁 기본 저장 경로")
    default_save_path = st.text_input(
        "기본 저장 폴더",
        value=camera_controller.save_directory
    )
    
    if st.button("📁 저장 경로 적용"):
        camera_controller.save_directory = default_save_path
        st.success("저장 경로가 설정되었습니다!")
    
    # 객체 탐지 설정
    st.markdown("#### 🔬 객체 탐지 설정")
    
    col1, col2 = st.columns(2)
    with col1:
        default_confidence = st.slider(
            "기본 신뢰도 임계값",
            0.1, 0.9, 0.5, 0.1,
            help="탐지를 위한 기본 신뢰도 설정"
        )
    
    with col2:
        use_gpu = st.checkbox(
            "GPU 가속 사용",
            value=camera_controller.performance_settings['use_gpu_acceleration'],
            help="가능한 경우 GPU를 사용하여 처리 속도 향상"
        )
    
    # 설정 저장
    if st.button("💾 설정 저장"):
        camera_controller.performance_settings.update({
            'confidence_threshold': default_confidence,
            'use_gpu_acceleration': use_gpu
        })
        st.success("설정이 저장되었습니다!")

def render_sidebar():
    """사이드바 렌더링"""
    with st.sidebar:
        st.header("시스템 정보")
        
        system_info = camera_controller.get_system_info()
        st.markdown("### 카메라 앱 지원")
        
        for app in system_info['supported_apps']:
            st.write(f"- **{app}**")
        
        if system_info['system'] == "Linux":
            st.markdown("#### 설치 명령어:")
            st.code("sudo apt install cheese", language="bash")
        
        # 객체 탐지 정보
        st.markdown("---")
        st.markdown("### 🔬 객체 탐지 정보")
        device_status = "GPU 가속" if system_info['cuda_available'] else "CPU 모드"
        st.write(f"**처리 장치**: {device_status}")
        st.write(f"**장치명**: {system_info['device_name']}")
        
        # 실시간 카메라 정보
        available_cameras = camera_controller.get_available_cameras()
        if available_cameras:
            st.markdown("### 사용 가능한 카메라")
            for cam in available_cameras:
                st.write(f"- **{cam['name']}** ({cam['resolution']})")
        
        # 실시간 탐지 상태
        if camera_controller.is_realtime_active():
            st.success("실시간 탐지 활성화")
        else:
            st.info("실시간 탐지 비활성화")
        
        st.markdown("---")
        st.markdown("### 🛠️ 기능")
        st.write("✅ 카메라 앱 제어")
        st.write("✅ 자동 파일 스캔")
        st.write("✅ 파일 미리보기")
        st.write("✅ 원하는 위치로 복사")
        st.write("✅ ZIP 압축 다운로드")
        st.write("✅ 날짜별 폴더 정리")
        st.write("🆕 순차 객체 탐지")
        st.write("🆕 사람/마스크 인식")
        st.write("🆕 세그멘테이션 분석")

def run_camera_ui_standalone():
    """독립 실행용 카메라 UI"""
    st.set_page_config(
        page_title="실시간 카메라 객체 탐지 시스템",
        page_icon="📹",
        layout="wide"
    )
    
    init_session_state()
    
    st.title("📹 실시간 카메라 객체 탐지 및 파일 관리 시스템")
    st.markdown("---")
    
    # 탭으로 기능 분리
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎮 카메라 제어", 
        "📁 파일 관리", 
        "🔬 객체 탐지",
        "📊 분석 결과",
        "⚙️ 설정"
    ])
    
    with tab1:
        render_camera_control_tab()
    
    with tab2:
        render_file_management_tab()
    
    with tab3:
        render_unified_detection_tab()
    
    with tab4:
        render_detection_analytics()
    
    with tab5:
        render_settings_tab()
    
    render_sidebar()
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "실시간 카메라 객체 탐지 시스템 | 촬영 → 실시간 분석 → 저장 → 관리"
        "</div>", 
        unsafe_allow_html=True
    )

def run_camera_ui():
    """기본 카메라 UI 함수 - 하위 호환성 유지"""
    run_camera_ui_standalone()