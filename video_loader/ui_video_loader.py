import streamlit as st
import time
from video_loader import video_loader
from PIL import Image
import io

def init_session_state():
    """세션 상태 초기화"""
    defaults = {
        'video_name': None,
        'video_bytes': None,
        'video_info': None,
        'thumbnail': None,
        'upload_history': [],
        'current_video_index': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def render_upload_sidebar():
    """사이드바 업로드 영역 렌더링"""
    st.sidebar.header("📂 영상 파일 불러오기")
    
    # 지원 형식 표시
    formats = video_loader.get_supported_formats()
    st.sidebar.info(f"지원 형식: {', '.join(formats)}")
    
    # 파일 업로드
    uploaded = st.sidebar.file_uploader(
        "동영상 선택",
        type=formats,
        accept_multiple_files=False,
        help="최대 500MB까지 업로드 가능합니다."
    )
    
    # 미리보기
    if uploaded is not None:
        st.sidebar.markdown("### 📺 미리보기")
        
        # 파일 정보 표시
        file_size = video_loader.format_file_size(uploaded.size)
        st.sidebar.write(f"**파일명**: {uploaded.name}")
        st.sidebar.write(f"**크기**: {file_size}")
        
        # 유효성 검사
        is_valid, message = video_loader.validate_file(uploaded)
        
        if is_valid:
            st.sidebar.success("✅ 유효한 파일")
            # 간단한 비디오 미리보기
            try:
                st.sidebar.video(uploaded)
            except Exception as e:
                st.sidebar.warning(f"미리보기 실패: {e}")
        else:
            st.sidebar.error(f"❌ {message}")
            uploaded = None  # 유효하지 않은 파일은 무효화
    
    # 첨부/취소 버튼
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        attach_clicked = st.button("첨부", key="attach_btn", use_container_width=True, type="primary")
    
    with col2:
        cancel_clicked = st.button("취소", key="cancel_btn", use_container_width=True)
    
    # 버튼 액션 처리
    if attach_clicked:
        if uploaded is None:
            st.sidebar.warning("먼저 파일을 선택해주세요.")
        else:
            with st.spinner("파일을 처리하는 중..."):
                # 파일 바이트 읽기
                video_bytes = uploaded.read()
                
                # 비디오 정보 추출
                video_info = video_loader.get_video_info(video_bytes)
                
                # 썸네일 추출
                thumbnail = video_loader.extract_thumbnail(video_bytes)
                
                # 세션 상태 업데이트
                st.session_state.video_name = uploaded.name
                st.session_state.video_bytes = video_bytes
                st.session_state.video_info = video_info
                st.session_state.thumbnail = thumbnail
                
                # 업로드 히스토리에 추가
                upload_record = {
                    'name': uploaded.name,
                    'size': uploaded.size,
                    'upload_time': time.time(),
                    'info': video_info
                }
                
                if 'upload_history' not in st.session_state:
                    st.session_state.upload_history = []
                
                # 중복 제거 (같은 이름의 파일)
                st.session_state.upload_history = [
                    record for record in st.session_state.upload_history 
                    if record['name'] != uploaded.name
                ]
                st.session_state.upload_history.append(upload_record)
                
                # 최근 10개만 유지
                if len(st.session_state.upload_history) > 10:
                    st.session_state.upload_history = st.session_state.upload_history[-10:]
            
            st.sidebar.success("파일이 성공적으로 첨부되었습니다!")
            st.rerun()
    
    if cancel_clicked:
        st.session_state.video_name = None
        st.session_state.video_bytes = None
        st.session_state.video_info = None
        st.session_state.thumbnail = None
        st.sidebar.info("선택된 파일이 취소되었습니다.")
        st.rerun()

def render_upload_history():
    """업로드 히스토리 렌더링"""
    if st.session_state.get('upload_history'):
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📁 최근 업로드")
        
        for i, record in enumerate(reversed(st.session_state.upload_history)):
            with st.sidebar.expander(f"{record['name'][:15]}..." if len(record['name']) > 15 else record['name']):
                st.write(f"**크기**: {video_loader.format_file_size(record['size'])}")
                
                if 'info' in record and 'duration' in record['info']:
                    duration = video_loader.format_duration(record['info']['duration'])
                    st.write(f"**재생시간**: {duration}")
                
                upload_time = time.strftime("%H:%M:%S", time.localtime(record['upload_time']))
                st.write(f"**업로드**: {upload_time}")
                
                if st.button(f"다시 로드", key=f"reload_{i}"):
                    # 이 기능은 실제로는 파일 바이트를 저장하고 있지 않으므로 
                    # 현재는 알림만 표시
                    st.warning("파일을 다시 선택해주세요.")

def render_main_monitor():
    """메인 모니터 영역 렌더링"""
    st.subheader("📺 영상 모니터")
    
    if st.session_state.get("video_bytes"):
        # 비디오 정보 표시
        if st.session_state.get("video_info"):
            render_video_info()
        
        # 비디오 플레이어
        st.markdown("### 🎬 비디오 플레이어")
        
        # 썸네일과 비디오를 탭으로 분리
        tab1, tab2 = st.tabs(["비디오 플레이어", "썸네일"])
        
        with tab1:
            try:
                st.video(st.session_state.video_bytes)
            except Exception as e:
                st.error(f"비디오 재생 오류: {e}")
        
        with tab2:
            if st.session_state.get("thumbnail") is not None:
                st.image(
                    st.session_state.thumbnail,
                    caption=f"썸네일: {st.session_state.video_name}",
                    use_container_width=True
                )
            else:
                st.info("썸네일을 생성할 수 없습니다.")
        
        # 비디오 액션 버튼들
        render_video_actions()
        
    else:
        # 비디오가 없을 때의 빈 상태
        render_empty_state()

def render_video_info():
    """비디오 정보 표시"""
    info = st.session_state.video_info
    
    if 'error' in info:
        st.error(f"비디오 정보를 가져올 수 없습니다: {info['error']}")
        return
    
    st.markdown("### 📊 비디오 정보")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "해상도",
            info.get('resolution', 'N/A'),
            help=f"가로 {info.get('width', 0)} x 세로 {info.get('height', 0)} 픽셀"
        )
    
    with col2:
        duration_str = video_loader.format_duration(info.get('duration', 0))
        st.metric("재생시간", duration_str)
    
    with col3:
        st.metric("프레임율", f"{info.get('fps', 0):.1f} FPS")
    
    with col4:
        st.metric("총 프레임", f"{info.get('frame_count', 0):,}개")
    
    # 추가 정보
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("파일 크기", f"{info.get('size_mb', 0):.1f} MB")
    
    with col2:
        # 예상 품질 평가
        quality = evaluate_video_quality(info)
        st.metric("예상 품질", quality)

def evaluate_video_quality(info):
    """비디오 품질 평가"""
    width = info.get('width', 0)
    height = info.get('height', 0)
    fps = info.get('fps', 0)
    
    if width >= 1920 and height >= 1080 and fps >= 30:
        return "고품질 (FHD+)"
    elif width >= 1280 and height >= 720 and fps >= 24:
        return "중품질 (HD)"
    elif width >= 640 and height >= 480:
        return "표준품질 (SD)"
    else:
        return "저품질"

def render_video_actions():
    """비디오 관련 액션 버튼들"""
    st.markdown("### 🛠️ 비디오 액션")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 새로고침", use_container_width=True):
            st.rerun()
    
    with col2:
        if st.button("📱 모바일 최적화", use_container_width=True):
            st.info("모바일 최적화 기능은 개발 중입니다.")
    
    with col3:
        if st.button("📊 상세 분석", use_container_width=True):
            show_detailed_analysis()
    
    with col4:
        if st.button("💾 임시 저장", use_container_width=True):
            save_to_temp()

def show_detailed_analysis():
    """상세 비디오 분석 표시"""
    if not st.session_state.get('video_info'):
        st.warning("비디오 정보가 없습니다.")
        return
    
    with st.expander("📈 상세 분석 결과", expanded=True):
        info = st.session_state.video_info
        
        # 기술적 분석
        st.markdown("#### 기술적 분석")
        
        # 비트레이트 추정
        size_mb = info.get('size_mb', 0)
        duration = info.get('duration', 1)
        estimated_bitrate = (size_mb * 8) / duration if duration > 0 else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**추정 비트레이트**: {estimated_bitrate:.1f} Mbps")
            st.write(f"**픽셀 수**: {info.get('width', 0) * info.get('height', 0):,} 픽셀")
        
        with col2:
            st.write(f"**화면비**: {get_aspect_ratio(info.get('width', 0), info.get('height', 0))}")
            st.write(f"**초당 데이터**: {estimated_bitrate/8:.1f} MB/s")
        
        # 권장사항
        st.markdown("#### 💡 권장사항")
        recommendations = generate_recommendations(info)
        for rec in recommendations:
            st.write(f"• {rec}")

def get_aspect_ratio(width, height):
    """화면비 계산"""
    if width == 0 or height == 0:
        return "N/A"
    
    from math import gcd
    common_divisor = gcd(width, height)
    ratio_w = width // common_divisor
    ratio_h = height // common_divisor
    
    return f"{ratio_w}:{ratio_h}"

def generate_recommendations(info):
    """비디오 기반 권장사항 생성"""
    recommendations = []
    
    fps = info.get('fps', 0)
    width = info.get('width', 0)
    duration = info.get('duration', 0)
    size_mb = info.get('size_mb', 0)
    
    if fps < 24:
        recommendations.append("프레임율이 낮습니다. 부드러운 재생을 위해 24fps 이상을 권장합니다.")
    
    if width < 1280:
        recommendations.append("해상도가 낮습니다. HD(1280x720) 이상의 해상도를 권장합니다.")
    
    if duration > 3600:  # 1시간 이상
        recommendations.append("긴 동영상입니다. 편집이나 분할을 고려해보세요.")
    
    if size_mb > 100:
        recommendations.append("파일 크기가 큽니다. 압축을 고려해보세요.")
    
    if not recommendations:
        recommendations.append("비디오 품질이 양호합니다.")
    
    return recommendations

def save_to_temp():
    """비디오를 임시 파일로 저장"""
    if not st.session_state.get('video_bytes') or not st.session_state.get('video_name'):
        st.warning("저장할 비디오가 없습니다.")
        return
    
    try:
        temp_path = video_loader.save_to_temp(
            st.session_state.video_bytes,
            st.session_state.video_name
        )
        st.success(f"임시 파일로 저장되었습니다: {temp_path}")
        
        # 다운로드 버튼 제공
        st.download_button(
            label="💾 다운로드",
            data=st.session_state.video_bytes,
            file_name=st.session_state.video_name,
            mime="video/mp4"
        )
        
    except Exception as e:
        st.error(f"저장 실패: {e}")

def render_empty_state():
    """비디오가 없을 때의 빈 상태 표시"""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #666;">
        <h3>🎬 비디오를 선택해주세요</h3>
        <p>왼쪽 사이드바에서 비디오 파일을 업로드하고 '첨부' 버튼을 클릭하세요.</p>
        <p style="font-size: 0.9em; margin-top: 2rem;">
            지원 형식: MP4, MOV, AVI, MKV, WEBM<br>
            최대 크기: 500MB
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_settings():
    """설정 영역 렌더링"""
    with st.expander("⚙️ 설정"):
        col1, col2 = st.columns(2)
        
        with col1:
            auto_play = st.checkbox("자동 재생", value=False)
            show_controls = st.checkbox("컨트롤 표시", value=True)
        
        with col2:
            loop_video = st.checkbox("반복 재생", value=False)
            mute_audio = st.checkbox("음소거", value=False)
        
        if st.button("🧹 임시 파일 정리"):
            video_loader.cleanup_temp_files()
            st.success("임시 파일이 정리되었습니다.")

def run_video_loader_ui():
    """비디오 로더 UI 실행"""
    st.set_page_config(
        page_title="Video Loader & Monitor", 
        page_icon="🎬", 
        layout="wide"
    )
    
    init_session_state()
    
    st.title("🎬 영상 모니터")
    st.markdown("---")
    
    # 사이드바
    render_upload_sidebar()
    render_upload_history()
    
    # 메인 영역
    render_main_monitor()
    
    # 설정
    render_settings()
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "영상 모니터 시스템 | 업로드 → 분석 → 모니터링"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    run_video_loader_ui()