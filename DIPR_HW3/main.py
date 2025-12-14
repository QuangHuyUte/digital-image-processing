import streamlit as st
import cv2
import numpy as np

# ==========================================
#          CẤU HÌNH TRANG
# ==========================================
st.set_page_config(layout="wide", page_title="DIP Homework Solver")

# ==========================================
#          HÀM XỬ LÝ ẢNH (BỎ CACHE)
# ==========================================

def create_gaussian_mask(shape, d0, type="lowpass"):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Tạo lưới tọa độ
    u = np.linspace(-ccol, cols - ccol - 1, cols)
    v = np.linspace(-crow, rows - crow - 1, rows)
    U, V = np.meshgrid(u, v)
    D_squared = U**2 + V**2
    
    # Tránh chia cho 0
    d0 = max(0.1, d0)
    
    # Gaussian Lowpass: H = exp(-D^2 / 2*D0^2)
    H = np.exp(-D_squared / (2 * (d0**2)))
    
    if type == "highpass":
        return 1 - H
    return H

def apply_fft_filter(img, H_mask):
    # 1. Biến đổi Fourier (FFT)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    
    # 2. Áp dụng bộ lọc (Nhân ma trận)
    fshift_filtered = fshift * H_mask
    
    # 3. Biến đổi ngược (IFFT)
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    
    # Lấy phần thực
    return np.real(img_back)

def process_display(img_float, mode):
    if mode == "Offset +128 (Nền xám - Sách GK)":
        # Cộng 128 để đẩy giá trị 0 lên mức xám giữa
        return np.clip(img_float + 128, 0, 255).astype(np.uint8)
    elif mode == "Normalize (0-255)":
        # Kéo dãn mức xám
        return cv2.normalize(img_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else: # Absolute
        return np.clip(np.abs(img_float), 0, 255).astype(np.uint8)

# ==========================================
#          GIAO DIỆN STREAMLIT
# ==========================================

st.title("🩻 Frequency Domain Image Processing")
st.write("Công cụ giải bài tập Xử lý ảnh số (Gaussian Filters).")

# --- SIDEBAR: Cài đặt hiển thị ---
with st.sidebar:
    st.header("Cài đặt hiển thị")
    view_mode = st.radio(
        "Chế độ xem kết quả:",
        ("Offset +128 (Nền xám - Sách GK)", "Normalize (0-255)", "Absolute Value"),
        index=0,
        help="Chọn 'Offset +128' để thấy kết quả giống trong đề bài (nền xám)."
    )

# --- TABS ---
tab1, tab2 = st.tabs(["👋 HW1: Hand X-Ray", "💾 HW2: PCB X-Ray"])

# ==========================================
#          TAB 1: HAND X-RAY
# ==========================================
with tab1:
    st.header("HW1: Lowpass + Highpass (Bandpass)")
    st.info("Bài tập yêu cầu: Lọc Lowpass (D0=25) sau đó lọc Highpass (D0=25).")

    col_upload, col_controls = st.columns([1, 2])
    
    with col_upload:
        uploaded_file_1 = st.file_uploader("Tải ảnh Bàn Tay (Hand)", type=['jpg', 'png', 'tif'], key="u1")

    if uploaded_file_1 is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_file_1.read()), dtype=np.uint8)
        img1 = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Resize nhẹ để chạy nhanh mượt (Max 512px)
        h, w = img1.shape
        if max(h, w) > 512:
            s = 512 / max(h, w)
            img1 = cv2.resize(img1, (0,0), fx=s, fy=s)

        with col_controls:
            c1, c2 = st.columns(2)
            d0_lp = c1.slider("Bước 1: Lowpass D0", 1, 100, 25, key="s1")
            d0_hp = c2.slider("Bước 2: Highpass D0", 1, 100, 25, key="s2")

        # --- XỬ LÝ (Real-time) ---
        # Tạo mask
        H_lp = create_gaussian_mask(img1.shape, d0_lp, "lowpass")
        H_hp = create_gaussian_mask(img1.shape, d0_hp, "highpass")
        
        # Kết hợp: H_total = H_low * H_high
        H_combined = H_lp * H_hp 
        
        # Chạy lọc
        res_float = apply_fft_filter(img1, H_combined)
        
        # Hậu xử lý hiển thị
        res_final = process_display(res_float, view_mode)

        # Hiển thị ảnh
        c_disp1, c_disp2 = st.columns(2)
        with c_disp1:
            st.write("**Ảnh Gốc**")
            st.image(img1, channels="GRAY", width="stretch")
        with c_disp2:
            st.write(f"**Kết quả (Lowpass {d0_lp} + Highpass {d0_hp})**")
            st.image(res_final, channels="GRAY", width="stretch")

# ==========================================
#          TAB 2: PCB X-RAY
# ==========================================
with tab2:
    st.header("HW2: Multi-pass Highpass")
    st.info("Bài tập yêu cầu: Lọc Gaussian Highpass nhiều lần (1, 10, 100 passes).")

    col_upload_2, col_controls_2 = st.columns([1, 2])
    
    with col_upload_2:
        uploaded_file_2 = st.file_uploader("Tải ảnh Bo Mạch (PCB)", type=['jpg', 'png', 'tif'], key="u2")

    if uploaded_file_2 is not None:
        # Đọc ảnh
        file_bytes = np.asarray(bytearray(uploaded_file_2.read()), dtype=np.uint8)
        img2 = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        
        # Resize nhẹ
        h, w = img2.shape
        if max(h, w) > 512:
            s = 512 / max(h, w)
            img2 = cv2.resize(img2, (0,0), fx=s, fy=s)

        with col_controls_2:
            c1, c2 = st.columns(2)
            d0_pcb = c1.slider("Gaussian Highpass D0", 1, 100, 30, key="s3")
            passes = c2.slider("Số lần lọc (Passes)", 1, 100, 1, key="s4")
            
            # Các nút chọn nhanh
            st.write("Chọn nhanh số lần lọc:")
            b1, b2, b3 = st.columns([1,1,1])
            if b1.button("1 Pass"): 
                passes = 1 
                # Lưu ý: Streamlit cần session state để update slider, nhưng ở đây ta ưu tiên chạy logic
                st.toast("Đã chọn 1 Pass. (Lưu ý: Slider có thể chưa cập nhật hình ảnh)")
            if b2.button("10 Passes"): passes = 10
            if b3.button("100 Passes"): passes = 100
            
        # --- XỬ LÝ (Real-time) ---
        # 1. Tạo mask gốc
        H_base = create_gaussian_mask(img2.shape, d0_pcb, "highpass")
        
        # 2. Lũy thừa mask (tương đương lọc N lần)
        # H_final = (H_base) ^ passes
        H_final = np.power(H_base, passes)
        
        # 3. Chạy lọc
        res_float_2 = apply_fft_filter(img2, H_final)
        
        # 4. Hiển thị
        res_final_2 = process_display(res_float_2, view_mode)

        c_disp3, c_disp4 = st.columns(2)
        with c_disp3:
            st.write("**Ảnh Gốc**")
            st.image(img2, channels="GRAY", width="stretch")
        with c_disp4:
            st.write(f"**Kết quả (D0={d0_pcb}, {passes} Passes)**")
            st.image(res_final_2, channels="GRAY", width="stretch")