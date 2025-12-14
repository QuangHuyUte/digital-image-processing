import streamlit as st
import cv2
import numpy as np

# ==========================================
#          HÀM TẠO ẢNH GIẢ LẬP (GENERATOR)
# ==========================================

def create_hw4_1_image():
    """Tạo ảnh nhị phân giống HW4-1: Chữ L và Hình tròn khuyết"""
    # Nền trắng (255), kích thước 300x600
    img = np.ones((300, 600), dtype=np.uint8) * 255
    
    # 1. Hình chữ L (Màu đen = 0)
    # Cạnh đứng
    cv2.rectangle(img, (50, 50), (130, 250), 0, -1)
    # Cạnh ngang
    cv2.rectangle(img, (50, 170), (230, 250), 0, -1)
    
    # 2. Hình tròn khuyết (Pacman)
    center = (450, 150)
    radius = 80
    cv2.circle(img, center, radius, 0, -1)
    # Cắt góc phần tư (vẽ tam giác trắng đè lên)
    # Tam giác từ tâm ra góc phải
    pts = np.array([center, (center[0] + radius, center[1] - radius), (center[0] + radius, center[1] + radius)], np.int32)
    # Vẽ hình chữ nhật trắng đè lên góc phần tư
    cv2.rectangle(img, center, (center[0] + radius, center[1] + radius), 255, -1)
    
    return img

def create_hw4_2_image():
    """Tạo ảnh nhị phân giống HW4-2: Sao, Đa giác, Khối tròn..."""
    img = np.ones((400, 800), dtype=np.uint8) * 255
    
    # 1. Ngôi sao 5 cánh (Góc trái trên)
    # Vẽ xấp xỉ bằng đa giác
    pts_star = np.array([[100, 50], [120, 110], [180, 110], [130, 150], 
                         [150, 210], [100, 170], [50, 210], [70, 150], 
                         [20, 110], [80, 110]], np.int32)
    cv2.fillPoly(img, [pts_star], 0)

    # 2. Ngôi sao 6 cánh (Giữa)
    center_hex = (350, 150)
    # Vẽ 2 tam giác ngược nhau
    tri1 = np.array([[350, 100], [300, 180], [400, 180]], np.int32)
    tri2 = np.array([[350, 200], [300, 120], [400, 120]], np.int32)
    cv2.fillPoly(img, [tri1], 0)
    cv2.fillPoly(img, [tri2], 0)

    # 3. Hình vuông bo góc (Phải trên) - Xấp xỉ
    cv2.rectangle(img, (550, 50), (700, 200), 0, -1)
    
    # 4. Bán nguyệt (Trái dưới)
    cv2.ellipse(img, (100, 300), (60, 60), 0, 0, 180, 0, -1)
    
    # 5. Hình chữ nhật bo tròn (Phải dưới)
    cv2.rectangle(img, (550, 280), (700, 350), 0, -1)
    cv2.circle(img, (550, 315), 35, 0, -1)
    cv2.circle(img, (700, 315), 35, 0, -1)

    return img

# ==========================================
#          CORE LOGIC (MORPHOLOGY)
# ==========================================

def get_diagonal_kernel():
    """
    Tạo Kernel đường chéo như đề bài HW4-1
    [1 0 0]
    [0 1 0]
    [0 0 1]
    """
    return np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.uint8)

def morphological_process(img, kernel, iterations, op_type):
    # OpenCV Morph hoạt động trên nền đen, vật thể trắng.
    # Ảnh đề bài là nền trắng, vật thể đen -> Cần đảo ngược (Invert) trước khi xử lý
    img_inv = cv2.bitwise_not(img)
    
    if op_type == "Erosion":
        res = cv2.erode(img_inv, kernel, iterations=iterations)
    elif op_type == "Dilation":
        res = cv2.dilate(img_inv, kernel, iterations=iterations)
    elif op_type == "Boundary Extraction":
        # Công thức: Boundary = A - (A erode B)
        erosion = cv2.erode(img_inv, kernel, iterations=1)
        res = img_inv - erosion
        
    # Đảo ngược lại về nền trắng, vật thể đen để hiển thị giống đề bài
    return cv2.bitwise_not(res)

# ==========================================
#          GIAO DIỆN STREAMLIT
# ==========================================

st.set_page_config(layout="wide", page_title="HW4: Morphology")

st.title("🔲 Morphological Image Processing")
st.markdown("Giải quyết bài tập **HW4-1 (Erosion/Dilation)** và **HW4-2 (Boundary Extraction)**.")

tab1, tab2 = st.tabs(["🧩 HW4-1: Specific Kernel", "🚩 HW4-2: Contour Extraction"])

# --- TAB 1: HW4-1 ---
with tab1:
    st.header("HW4-1: Erosion & Dilation with Diagonal Kernel")
    st.info("Yêu cầu: Tạo ảnh và xử lý bằng Kernel đường chéo.")
    
    # 1. Tạo ảnh
    img1 = create_hw4_1_image()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.write("### Kernel (Structuring Element)")
        st.latex(r'''
        B = \begin{bmatrix} 
        1 & 0 & 0 \\ 
        0 & 1 & 0 \\ 
        0 & 0 & 1 
        \end{bmatrix}
        ''')
        kernel_diag = get_diagonal_kernel()
        iter_num = st.slider("Số lần lặp (Iterations)", 1, 10, 1)
        
    with col2:
        # Xử lý
        eroded = morphological_process(img1, kernel_diag, iter_num, "Erosion")
        dilated = morphological_process(img1, kernel_diag, iter_num, "Dilation")
        
        # Hiển thị
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img1, caption="Original Input", width=None, use_container_width=True)
        with c2:
            st.image(eroded, caption=f"Erosion ({iter_num} iter)", width=None, use_container_width=True)
        with c3:
            st.image(dilated, caption=f"Dilation ({iter_num} iter)", width=None, use_container_width=True)
            
    st.success("""
    **Nhận xét:**
    - Vì Kernel là đường chéo (góc trên trái xuống dưới phải), khi **Erosion**, vật thể bị co lại theo hướng chéo.
    - Khi **Dilation**, vật thể giãn nở ra theo hướng chéo, tạo cảm giác bị "kéo dãn" (shear) hoặc mọc thêm gai ở góc chéo.
    """)

# --- TAB 2: HW4-2 ---
with tab2:
    st.header("HW4-2: Extract Borders (Logic Visualization)")
    st.info("Công thức: Border = Original - Erode(Original)")
    
    # 1. Tạo ảnh
    img2 = create_hw4_2_image()
    
    # 2. Thanh kéo điều chỉnh độ dày (Logic quan sát)
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        k_size = st.slider("Độ dày đường biên (Kernel Size)", min_value=3, max_value=21, value=3, step=2, help="Kernel càng to, ảnh bị ăn mòn càng nhiều -> Viền càng dày")
    
    # 3. Xử lý từng bước để hiển thị
    # B1: Đảo màu (để OpenCV hiểu vật thể là màu trắng)
    img_inv = cv2.bitwise_not(img2)
    
    # B2: Tạo Kernel
    kernel = np.ones((k_size, k_size), np.uint8)
    
    # B3: Thực hiện Erosion (Co nhỏ vật thể)
    eroded_inv = cv2.erode(img_inv, kernel, iterations=1)
    
    # B4: Trừ ảnh (Gốc - Co nhỏ = Biên)
    border_inv = img_inv - eroded_inv
    
    # B5: Đảo lại màu để hiển thị (Nền trắng, vật đen)
    img_display_orig = img2
    img_display_erode = cv2.bitwise_not(eroded_inv)
    img_display_border = cv2.bitwise_not(border_inv)
    
    # 4. Hiển thị 3 cột để thấy rõ Logic
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.image(img_display_orig, caption="1. Ảnh Gốc (A)", use_container_width=True)
        st.caption("Kích thước vật thể chuẩn.")
        
    with c2:
        st.image(img_display_erode, caption=f"2. Ảnh bị Co (A \u2296 B)", use_container_width=True)
        st.caption(f"Vật thể bị gọt bớt {k_size//2} pixel mỗi chiều.")
        
    with c3:
        st.image(img_display_border, caption="3. Kết quả (A - Co)", use_container_width=True)
        st.caption("Phần chênh lệch giữ lại chính là Biên.")
        
    # Giải thích Logic
    st.success(f"""
    **Giải thích Logic:**
    Khi bạn kéo thanh trượt lên **{k_size}**:
    1. Vật thể ở hình (2) bị co nhỏ lại đáng kể so với hình (1).
    2. Phép trừ **(Hình 1) - (Hình 2)** sẽ để lại một khoảng trống lớn hơn.
    3. Kết quả là đường biên ở hình (3) trở nên **dày hơn**.
    """)