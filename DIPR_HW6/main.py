import streamlit as st
import cv2
import numpy as np
from PIL import Image

# 1. Cấu hình trang Streamlit
st.set_page_config(page_title="Nhận diện khuôn mặt", page_icon="📷")

st.title("📷 Ứng dụng Nhận diện Khuôn mặt")
st.markdown(
    """
    Ứng dụng sử dụng **Haar Cascade Classifiers** để phát hiện khuôn mặt từ Webcam.
    """
)

# 2. Tải mô hình Haar Cascade
# Sử dụng @st.cache_resource để chỉ tải model một lần, giúp app chạy nhanh hơn
@st.cache_resource
def load_cascade():
    # OpenCV đã tích hợp sẵn các file xml, ta chỉ cần gọi đường dẫn
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return face_cascade

face_cascade = load_cascade()

# 3. Giao diện điều khiển
col1, col2 = st.columns(2)
with col1:
    run = st.checkbox('Bật/Tắt Webcam', value=False)
with col2:
    scale_factor = st.slider("Độ nhạy (Scale Factor)", 1.1, 2.0, 1.1, 0.1)
    min_neighbors = st.slider("Độ chính xác (Min Neighbors)", 3, 10, 5)

# Khung hiển thị hình ảnh (Placeholder)
FRAME_WINDOW = st.image([])

# 4. Logic xử lý Webcam
camera = cv2.VideoCapture(0)  # 0 là ID của webcam mặc định

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Không thể truy cập Webcam. Vui lòng kiểm tra lại kết nối!")
        break
    
    # Lật ngược ảnh cho giống gương (tùy chọn)
    frame = cv2.flip(frame, 1)

    # Chuyển sang ảnh xám (Grayscale) để tăng tốc độ nhận diện
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 
    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(30, 30)
    )

    # Vẽ hình chữ nhật quanh khuôn mặt
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # OpenCV dùng hệ màu BGR, Streamlit dùng RGB -> Cần chuyển đổi
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Hiển thị lên Streamlit
    FRAME_WINDOW.image(frame_rgb)

else:
    # Khi bỏ tích checkbox, dừng camera và thông báo
    camera.release()
    st.info("Hãy tích vào 'Bật/Tắt Webcam' để bắt đầu.")