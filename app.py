import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from streamlit_canvas import st_canvas

# Tối ưu giao diện Streamlit rộng rãi, hiện đại
st.set_page_config(
    page_title="AI Animal Sketch Guesser",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Thêm hiệu ứng CSS để làm đẹp giao diện
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stTitle { font-family: 'Helvetica Neue', sans-serif; font-weight: 800; color: #1E293B; text-align: center; }
    .prediction-card { background-color: #ffffff; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .animal-name { font-size: 32px; font-weight: bold; color: #4F46E5; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# Tải mô hình AI và lưu vào bộ nhớ cache để app chạy mượt hơn
@st.cache_resource
def load_ai_model():
    return load_model("animal_sketch_model.h5")

try:
    model = load_ai_model()
except Exception as e:
    st.error("Không tìm thấy file 'animal_sketch_model.h5'. Hãy đảm bảo bạn đã đặt file này cùng thư mục với app.py!")
    st.stop()

classes = ["cat", "dog", "fish", "bird", "rabbit", "lion", "tiger", "elephant", "monkey", "horse"]
emoji_dict = {"cat":"🐱", "dog":"🐶", "fish":"🐟", "bird":"🐦", "rabbit":"🐰", "lion":"🦁", "tiger":"🐯", "elephant":"🐘", "monkey":"🐵", "horse":"🐴"}

# Tiêu đề chính của Ứng dụng
st.markdown("<h1 class='stTitle'>🐾 AI Drawing Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748B;'>Vẽ một con vật bất kỳ và để trí tuệ nhân tạo dự đoán kết quả trong tích tắc!</p>", unsafe_allow_html=True)
st.write("---")

# Chia giao diện làm 2 cột cân đối: Bên trái vẽ - Bên phải hiện kết quả
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown("### 🎨 Bảng Vẽ Của Bạn")
    st.caption("Mẹo: Vẽ nét rõ ràng, ở chính giữa khung để AI đoán chuẩn nhất.")
    
    # Tạo khung vẽ bo góc, chuyên nghiệp
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0)", 
        stroke_width=10,                      # Độ dày nét vẽ tối ưu cho nét vẽ tay
        stroke_color="#000000",
        background_color="#FFFFFF",
        update_streamlit=True,
        height=380,
        width=380,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("### 🧠 AI Phân Tích")
    
    if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
        img_raw = canvas_result.image_data
        
        # Tiền xử lý ảnh theo chuẩn Quick Draw
        img = cv2.cvtColor(img_raw, cv2.COLOR_RGBA2RGB)
        img = cv2.resize(img, (28, 28))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = 255 - img
        img = img.astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)

        # Dự đoán
        prediction = model.predict(img, verbose=0)[0]
        index = np.argmax(prediction)
        animal = classes[index]
        confidence = float(np.max(prediction) * 100)
        emoji = emoji_dict.get(animal, "🐾")

        # Hiển thị kết quả chính bằng Card CSS sang xịn mịn
        st.markdown(f"""
            <div class="prediction-card">
                <p style="margin:0; color:#64748B; font-weight:600;">KẾT QUẢ DỰ ĐOÁN CAO NHẤT</p>
                <span class="animal-name">{emoji} {animal}</span>
                <h2 style="color:#10B981; margin:0;">Độ chính xác: {confidence:.2f}%</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Hiển thị thanh tiến trình (Progress Bar) cho top các con vật
        st.markdown("<br><b>📊 Xác suất chi tiết các loài:</b>", unsafe_allow_html=True)
        # Sắp xếp các kết quả từ cao xuống thấp để hiển thị đẹp hơn
        sorted_indices = np.argsort(prediction)[::-1]
        for i in sorted_indices[:4]: # Chỉ hiện Top 4 con vật có phần trăm cao nhất
            c = classes[i]
            prob = prediction[i]
            st.write(f"{emoji_dict[c]} **{c.capitalize()}**")
            st.progress(float(prob))
            
    else:
        # Trạng thái chờ khi người dùng chưa vẽ
        st.info("👋 Đang đợi bạn đặt bút vẽ... Hãy vẽ một con vật ở khung bên trái nhé!")
        st.markdown("""
        **Danh sách con vật AI đã học:**
        `Mèo`, `Chó`, `Cá`, `Chim`, `Thỏ`, `Sư tử`, `Hổ`, `Voi`, `Khỉ`, `Ngựa`.
        """)
