import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_dataset, preprocess_data
from model import train_softmax, evaluate_model

# Cấu hình trang
st.set_page_config(page_title="Đồ án: Softmax Regression", layout="wide")

st.title("Đồ án Machine Learning: Thuật toán Softmax Regression")
st.markdown("Hệ thống demo phân loại đa lớp (Multi-class Classification) bằng Softmax.")

# --- SIDEBAR: CẤU HÌNH ---
st.sidebar.header("Cấu hình Mô hình")

# 1. Chọn bộ dữ liệu
dataset_name = st.sidebar.selectbox(
    "Chọn bộ dữ liệu thực nghiệm:",
    ("Iris (Hoa Diên Vĩ - 3 Lớp)", "Wine (Chất lượng rượu - 3 Lớp)", "Digits (Chữ số - 10 Lớp)")
)

# 2. Chọn tỷ lệ chia tập Test
test_size = st.sidebar.slider("Tỷ lệ tập Test (Test Size):", 0.1, 0.5, 0.2, 0.05)

# 3. Tham số C (Nghịch đảo của mức độ Regularization)
c_param = st.sidebar.slider("Tham số C (Inverse of Regularization):", 0.01, 10.0, 1.0, 0.1)

# --- XỬ LÝ DỮ LIỆU ---
X, y, target_names = get_dataset(dataset_name)

st.subheader(f"Thông tin bộ dữ liệu: {dataset_name.split(' ')[0]}")
st.write(f"- **Số lượng mẫu:** {X.shape[0]}")
st.write(f"- **Số lượng đặc trưng (Features):** {X.shape[1]}")
st.write(f"- **Số lớp phân loại (Classes):** {len(target_names)} lớp ({', '.join(target_names)})")

# Hiển thị 5 dòng dữ liệu đầu tiên
with st.expander("Xem trước dữ liệu (Raw Data)"):
    st.dataframe(X.head())

# Tiền xử lý
X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size)

# --- HUẤN LUYỆN VÀ ĐÁNH GIÁ ---
if st.sidebar.button("Huấn luyện Mô hình"):
    with st.spinner("Đang huấn luyện Softmax..."):
        # Train
        model = train_softmax(X_train, y_train, c_param)
        
        # Đánh giá
        acc, report_df, cm = evaluate_model(model, X_test, y_test, target_names)
        
        st.success("Huấn luyện thành công!")
        
        st.markdown("---")
        st.header("Kết quả Đánh giá (Chương 5)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric(label="Độ chính xác (Accuracy)", value=f"{acc * 100:.2f}%")
            st.markdown("**Báo cáo phân loại (Classification Report):**")
            st.dataframe(report_df.style.highlight_max(axis=0, subset=['f1-score']))
            
        with col2:
            st.markdown("**Ma trận nhầm lẫn (Confusion Matrix):**")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.ylabel('Thực tế (Actual)')
            plt.xlabel('Dự đoán (Predicted)')
            st.pyplot(fig)
else:
    st.info("Hãy điều chỉnh thông số ở thanh bên trái và nhấn 'Huấn luyện Mô hình' để xem kết quả.")