import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# ==========================================
#          CẤU HÌNH & CSS
# ==========================================
st.set_page_config(layout="wide", page_title="DIPR HW5")

# Tắt cảnh báo spam của Streamlit
import logging
logging.getLogger("streamlit.runtime.media_file_storage").setLevel(logging.ERROR)

# ==========================================
#          HÀM XỬ LÝ NÂNG CAO
# ==========================================

def adjust_brightness(image, value):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if value >= 0:
        v = cv2.add(v, value)
    else:
        v = cv2.subtract(v, abs(value))
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def safe_display_image(placeholder, image, channels="RGB"):
    """Hàm hiển thị ảnh an toàn, tránh lỗi Missing File"""
    try:
        if channels == "BGR":
            # Tự convert sang RGB để Streamlit đỡ phải làm, giảm tải bộ nhớ
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # width='stretch' thay cho use_container_width (chuẩn mới)
        placeholder.image(image, channels="RGB", width="stretch") 
    except Exception:
        pass # Bỏ qua lỗi nếu render quá nhanh

def process_video_safe(video_path, thresh, min_area, history):
    """Hàm xử lý video tối ưu cho Streamlit"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        st.error("Không thể đọc file video.")
        return

    # Khởi tạo MOG2
    back_sub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=thresh, detectShadows=True)
    
    # Tạo placeholder rỗng để chứa video
    st_frame = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Nếu hết video, quay lại từ đầu (Loop) để dễ chỉnh sửa
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
            
        # 1. Trừ nền
        fg_mask = back_sub.apply(frame)
        
        # 2. Xử lý Mask
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8), iterations=1)
        fg_mask = cv2.dilate(fg_mask, np.ones((3,3),np.uint8), iterations=2)
        
        # 3. Tìm biên
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res_frame = frame.copy()
        
        cnt_found = 0
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                cnt_found += 1
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(res_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(res_frame, "Motion", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        
        # 4. Hiển thị (Resize nhẹ để chạy mượt hơn)
        h, w = frame.shape[:2]
        dim = (400, int(h * (400/w)))
        
        vis_orig = cv2.resize(res_frame, dim)
        vis_mask = cv2.resize(cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR), dim)
        
        combined = np.hstack((vis_orig, vis_mask))
        
        # Gọi hàm hiển thị an toàn
        safe_display_image(st_frame, combined, channels="BGR")

    cap.release()

def advanced_segmentation(image, blur_k, thresh_val, use_otsu, bg_mode, min_area, use_saturation=False):
    # 1. Chọn kênh màu
    if use_saturation:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        process_channel = hsv[:,:,1] 
    else:
        process_channel = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Làm mờ
    k = blur_k if blur_k % 2 == 1 else blur_k + 1
    blurred = cv2.GaussianBlur(process_channel, (k, k), 0)
    
    # 3. Phân ngưỡng
    if use_saturation:
        base_type = cv2.THRESH_BINARY
    else:
        if bg_mode == "Nền Sáng (Vật thể tối)":
            base_type = cv2.THRESH_BINARY_INV
        else:
            base_type = cv2.THRESH_BINARY

    if use_otsu:
        thresh_type = base_type + cv2.THRESH_OTSU
        final_thresh_val, binary = cv2.threshold(blurred, 0, 255, thresh_type)
    else:
        final_thresh_val = thresh_val
        _, binary = cv2.threshold(blurred, thresh_val, 255, base_type)
        
    # 4. Morphology
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    morph = cv2.dilate(morph, kernel, iterations=1)
    
    # 5. Đếm
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    res_img = image.copy()
    count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            count += 1
            cv2.drawContours(res_img, [cnt], -1, (0, 255, 0), 3)
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                cv2.putText(res_img, f"#{count}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
                cv2.putText(res_img, f"#{count}", (cx-10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return binary, res_img, count

# ==========================================
#          GIAO DIỆN CHÍNH
# ==========================================

st.title("🎓 Giải Bài Tập chapter 5")
tabs = st.tabs(["🎥 Câu 1: Video Motion", "📚 Câu 2 & 3: Lý thuyết", "🔦 Câu 4: Ánh sáng", "🍎 Câu 5: Đếm quả"])

# --- TAB 1: VIDEO (ĐÃ SỬA LỖI MẤT HÌNH) ---
with tabs[0]:
    st.header("Câu 1: Trích xuất chuyển động")
    
    # Cấu hình thanh trượt TRƯỚC KHI load video
    c1, c2, c3 = st.columns(3)
    v_thresh = c1.slider("Độ nhạy (Threshold)", 10, 200, 50, key="v1", help="Kéo lên nếu bị nhiễu quá nhiều")
    v_min_area = c2.slider("Lọc nhiễu (Min Area)", 10, 2000, 500, key="v2")
    v_hist = c3.slider("Lịch sử (History)", 100, 1000, 500, key="v3")

    vf = st.file_uploader("Upload Video (mp4/avi)", type=["mp4", "avi", "mov"])
    
    # Logic lưu file tạm vào Session State để không bị mất khi Rerun
    if vf:
        if 'video_path' not in st.session_state:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(vf.read())
            tfile.close()
            st.session_state.video_path = tfile.name # Lưu đường dẫn
        
        # Nút chạy riêng
        run_video = st.checkbox("▶️ Chạy Video (Tự động Loop)", value=True)
        
        if run_video:
            # Gọi hàm xử lý với đường dẫn đã lưu
            try:
                process_video_safe(st.session_state.video_path, v_thresh, v_min_area, v_hist)
            except Exception as e:
                # Nếu file lỗi thì reset session
                if 'video_path' in st.session_state:
                    del st.session_state.video_path
                st.error("Có lỗi xảy ra, vui lòng upload lại video.")

# --- TAB 2 & 3 ---
with tabs[1]:
    st.markdown("### Câu 2: Công thức Ngưỡng tối ưu")
    st.latex(r"T = \frac{\mu_1 + \mu_2}{2} + \frac{\sigma^2}{\mu_1 - \mu_2} \ln\left(\frac{p_2}{p_1}\right)")
    st.markdown("### Câu 3: Hạn chế Histogram")
    st.write("- Mất thông tin không gian.\n- Nhiễu làm sai lệch đỉnh.\n- Khó khăn khi ánh sáng không đều.")

# --- TAB 4: ÁNH SÁNG & 2 VẬT THỂ ---
with tabs[2]:
    st.header("Câu 4: Phân đoạn 2 vật thể")
    
    uf_c4 = st.file_uploader("Upload ảnh vật thể (Câu 4)", type=['jpg','png'])
    if uf_c4:
        fb = np.asarray(bytearray(uf_c4.read()), dtype=np.uint8)
        img_c4 = cv2.imdecode(fb, cv2.IMREAD_COLOR)
        
        st.divider()
        c_ctrl1, c_ctrl2 = st.columns(2)
        
        with c_ctrl1:
            st.markdown("**1. Giả lập Ánh sáng**")
            brightness = st.slider("Độ sáng", -100, 100, 0, key="b4")
            img_c4_adj = adjust_brightness(img_c4, brightness)
            
        with c_ctrl2:
            st.markdown("**2. Cấu hình Phân đoạn**")
            use_otsu_c4 = st.checkbox("Auto Otsu", value=True, key="otsu4")
            thresh_c4 = st.slider("Ngưỡng thủ công", 0, 255, 100, disabled=use_otsu_c4, key="th4")
            bg_mode_c4 = st.selectbox("Kiểu nền:", ["Nền Sáng (Vật thể tối)", "Nền Tối (Vật thể sáng)"], key="bg4")
            min_area_c4 = st.slider("Min Area", 0, 5000, 100, key="ma4")

        bin_c4, res_c4, cnt_c4 = advanced_segmentation(
            img_c4_adj, 5, thresh_c4, use_otsu_c4, bg_mode_c4, min_area_c4
        )
        
        col1, col2 = st.columns(2)
        col1.write("Ảnh Input (Đã chỉnh sáng)")
        safe_display_image(col1, img_c4_adj, channels="BGR")
        
        col2.write(f"Kết quả: {cnt_c4} vật thể")
        safe_display_image(col2, res_c4, channels="BGR")
        
        with st.expander("Xem Mask đen trắng"):
            safe_display_image(st, bin_c4, channels="GRAY")

# --- TAB 5: ĐẾM QUẢ ---
with tabs[3]:
    st.header("Câu 5: Đếm trái cây trong ảnh")
    
    uf5 = st.file_uploader("Upload ảnh trái cây", type=['jpg','png'], key="u5")
    
    # Load ảnh Demo nếu chưa upload
    if not uf5:
        st.info("Đang dùng ảnh Demo (Upload ảnh để test thật)")
        img5 = np.ones((400, 600, 3), dtype=np.uint8)
        img5[:] = (200, 230, 250) 
        cv2.circle(img5, (100, 100), 40, (50, 50, 200), -1) 
        cv2.ellipse(img5, (450, 120), (80, 25), 30, 0, 360, (0, 255, 255), -1) 
        cv2.circle(img5, (250, 250), 30, (50, 200, 50), -1) 
    else:
        fb = np.asarray(bytearray(uf5.read()), dtype=np.uint8)
        img5 = cv2.imdecode(fb, cv2.IMREAD_COLOR)

    if img5 is not None:
        st.divider()
        col_L, col_R = st.columns([1, 2])
        
        with col_L:
            st.subheader("🔧 Bộ lọc")
            use_sat = st.checkbox("✅ Dùng kênh Màu Sắc (Saturation)", value=True, 
                                  help="Bật cái này để tách quả vàng trên nền trắng/kem.")
            
            st.markdown("---")
            use_otsu_5 = st.checkbox("Auto Otsu", value=True, key="otsu5")
            thresh_5 = st.slider("Ngưỡng (Threshold)", 0, 255, 60, disabled=use_otsu_5, 
                                 help="Nếu dùng Saturation: Kéo thấp (20-60).")
            
            min_area_5 = st.slider("Lọc nhiễu nhỏ", 10, 2000, 200, key="ma5")
            
            bg_mode_5 = "Nền Sáng (Vật thể tối)"
            if not use_sat:
                bg_mode_5 = st.selectbox("Kiểu nền:", ["Nền Sáng (Vật thể tối)", "Nền Tối (Vật thể sáng)"], key="bg5")

        bin_5, res_5, count_5 = advanced_segmentation(
            img5, 5, thresh_5, use_otsu_5, bg_mode_5, min_area_5, use_saturation=use_sat
        )
        
        with col_R:
            st.metric("Kết quả đếm:", f"{count_5} Quả")
            safe_display_image(st, res_5, channels="BGR")
            
            with st.expander("🔍 Debug: Tại sao đếm sai?"):
                st.write("Ảnh nhị phân (Trắng = Quả)")
                safe_display_image(st, bin_5, channels="GRAY")