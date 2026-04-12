import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

from data_loader import get_dataset
from data_preprocessing import clean_data
from model import train_softmax, evaluate_model, cross_validate_model, diagnose_overfitting_underfitting, diagnose_imbalanced_data, diagnose_cv_stability

# Hằng số cấu hình
OVERFITTING_GAP_THRESHOLD = 0.10
UNDERFITTING_ACC_THRESHOLD = 0.70
IMBALANCE_F1_THRESHOLD = 0.05
CV_STABILITY_THRESHOLD = 0.05

st.set_page_config(page_title="Softmax Regression", layout="wide")

st.title("Đồ án Machine Learning: Softmax Regression")
st.markdown("Hệ thống phân loại đa lớp với xử lý dữ liệu thực tế và đánh giá ổn định.")

if 'last_data_source' not in st.session_state:
    st.session_state.last_data_source = None
if 'last_dataset_name' not in st.session_state:
    st.session_state.last_dataset_name = None
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None

st.sidebar.header("Cấu hình hệ thống")

data_source = st.sidebar.radio("Nguồn dữ liệu:", 
                               ("Thư viện Sklearn (Sạch)", "Tải lên file CSV"))

data_source_changed = False
if data_source != st.session_state.last_data_source:
    data_source_changed = True
    st.session_state.model = None
    st.session_state.last_data_source = data_source

if data_source == "Thư viện Sklearn (Sạch)":
    dataset_name = st.sidebar.selectbox(
        "Chọn bộ dữ liệu:", 
        ("Iris (Hoa Diên Vĩ - 3 Lớp)", "Wine (Chất lượng rượu - 3 Lớp)", "Digits (Chữ số - 10 Lớp)")
    )
    
    if dataset_name != st.session_state.last_dataset_name:
        st.session_state.model = None
        st.session_state.last_dataset_name = dataset_name
        if st.session_state.last_dataset_name is not None:
            st.sidebar.warning("Dataset đã thay đổi! Mô hình cũ đã xóa. Hãy train lại.")
    
    X, y, target_names = get_dataset(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Chọn file CSV", type="csv")
    if uploaded_file is not None:
        # Kiểm tra thay đổi file
        if uploaded_file.name != st.session_state.last_uploaded_file_name:
            st.session_state.model = None
            st.session_state.last_uploaded_file_name = uploaded_file.name
            if st.session_state.last_uploaded_file_name is not None:
                st.sidebar.warning("File đã thay đổi! Mô hình cũ đã xóa. Hãy train lại.")
        
        df_raw = pd.read_csv(uploaded_file)
        st.info(f"Đã tải: {df_raw.shape[0]:,} dòng, {df_raw.shape[1]} cột")
        
        with st.expander("Xem dữ liệu gốc"):
            st.dataframe(df_raw.head(10))
        
        target_col = st.sidebar.selectbox("Chọn cột nhãn (target):", df_raw.columns, 
                                          index=len(df_raw.columns)-1)
        
        unique_target_values = df_raw[target_col].nunique()
        if unique_target_values > 50:
            st.warning(f"""
            Cảnh báo: Cột "{target_col}" có {unique_target_values} giá trị duy nhất!           
            Điều này có thể không phải cột nhãn (target) mà là cột đặc trưng (feature).
            Vui lòng chọn lại cột target khác
            """)
            st.stop()
        
        try:
            X, y, target_names = clean_data(df_raw, target_col)
            st.success(f"Làm sạch thành công: {X.shape[0]:,} mẫu, {X.shape[1]} đặc trưng")
        except Exception as e:
            st.error(f"Lỗi xử lý dữ liệu: {str(e)}")
            st.stop()
    else:
        st.warning("Vui lòng tải file CSV để tiếp tục.")
        st.stop()

test_size = st.sidebar.slider("Tỷ lệ tập Test:", 0.1, 0.5, 0.2, step=0.05)
c_param = st.sidebar.slider("Tham số C (Regularization):", 0.01, 100.0, 1.0, step=0.1)

st.subheader("Thông tin dataset")
num_classes = len(target_names)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Tổng số mẫu", f"{X.shape[0]:,}")
    st.metric("Số đặc trưng", X.shape[1])
with col2:
    st.metric("Số lớp", num_classes)
with col3:
    st.metric("Tỷ lệ Test", f"{test_size*100:.0f}%")

st.write(f"Danh sách lớp: {', '.join(map(str, target_names))}")

st.subheader("Phân phối lớp")
class_counts = pd.Series(y).value_counts().sort_index()
imbalance_ratio = class_counts.max() / class_counts.min()

col_dist1, col_dist2 = st.columns([3, 1])
with col_dist1:
    fig, ax = plt.subplots()
    class_counts.plot(kind='bar', ax=ax, color='skyblue')
    ax.set_xlabel("Lớp")
    ax.set_ylabel("Số lượng mẫu")
    plt.xticks(rotation=0)
    st.pyplot(fig)

with col_dist2:
    st.metric("Imbalance Ratio", f"{imbalance_ratio:.2f}:1", 
              help="Tỷ lệ lớp lớn nhất / lớp nhỏ nhất. > 2.0 là có lệch đáng kể.")
    if imbalance_ratio > 2.0:
        st.warning("Dữ liệu bị lệch lớp khá mạnh")
    else:
        st.success("Dữ liệu khá cân bằng")

with st.expander("Xem dữ liệu sau xử lý"):
    st.dataframe(X.head())

if st.sidebar.button("Bắt đầu huấn luyện", type="primary"):
        with st.spinner("Đang huấn luyện và đánh giá..."):
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            
            # Scaling trong Single Train mode
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            model = train_softmax(X_train, y_train, c_param)
            acc, report_df, cm, brier_score = evaluate_model(model, X_test, y_test, target_names)
            train_acc = model.score(X_train, y_train)
            
            # Lưu mô hình, scaler và dữ liệu test vào session state
            st.session_state.model = model
            st.session_state.scaler = scaler
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test

        st.success("Huấn luyện hoàn tất!")

        st.markdown("---")
        st.header("Kết quả Train")

        col_a, col_b = st.columns(2)
        with col_a: st.metric("Train Accuracy", f"{train_acc*100:.2f}%")
        with col_b: st.metric("Test Accuracy", f"{acc*100:.2f}%")

        gap = train_acc - acc
        y_pred = model.predict(X_test)
        
        # Lấy Macro F1 từ Classification Report
        macro_f1 = report_df.loc['macro avg', 'f1-score']

        st.subheader("Chẩn đoán Overfitting / Underfitting")
        
        # Gọi hàm chẩn đoán với 3 chỉ số: Train Acc, Test Acc, Macro F1
        diagnosis = diagnose_overfitting_underfitting(
            train_acc, acc, macro_f1, 
            threshold_gap=OVERFITTING_GAP_THRESHOLD, 
            threshold_f1=0.65
        )
        
        # Hiển thị kết quả chẩn đoán
        if diagnosis['status'] == 'OVERFITTING':
            st.warning(f"""
            {diagnosis['message']}  
            → {diagnosis['recommendation']}
            """)
        elif diagnosis['status'] == 'OVERFITTING_LECH_LOP':
            st.error(f"""
            {diagnosis['message']}  
            → {diagnosis['recommendation']}
            """)
        elif diagnosis['status'] == 'UNDERFITTING':
            st.error(f"""
            {diagnosis['message']}  
            → {diagnosis['recommendation']}
            """)
        else:  # STABLE
            st.success(f"""
        {diagnosis['message']}  
        → {diagnosis['recommendation']}
        """)


        # Hiển thị báo cáo chi tiết và ma trận nhầm lẫn
        col_rep, col_cm = st.columns([1, 1.6])
        with col_rep:
            st.subheader("Classification Report")
            st.dataframe(report_df.style.format("{:.4f}"))
        with col_cm:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', 
                        xticklabels=target_names, yticklabels=target_names)
            plt.ylabel('Thực tế')
            plt.xlabel('Dự đoán')
            plt.tight_layout()
            st.pyplot(fig)

        # Nhóm 3: Ảnh hưởng mất cân bằng dữ liệu
        st.subheader("Chỉ số Macro F1 vs Weighted F1")
        
        macro_f1 = report_df.loc['macro avg', 'f1-score']
        weighted_f1 = report_df.loc['weighted avg', 'f1-score']
        
        # Gọi hàm chẩn đoán từ model.py
        imbalance_diagnosis = diagnose_imbalanced_data(macro_f1, weighted_f1, IMBALANCE_F1_THRESHOLD)
        
        col_f1_1, col_f1_2, col_f1_3 = st.columns(3)
        with col_f1_1:
            st.metric("Macro F1", f"{macro_f1:.4f}", 
                     help="Trung bình cộng F1 của tất cả lớp")
        with col_f1_2:
            st.metric("Weighted F1", f"{weighted_f1:.4f}", 
                     help="Trung bình có trọng số theo số mẫu (ưu tiên lớp đa số)")
        with col_f1_3:
            st.metric("Gap", f"{imbalance_diagnosis['gap']:.4f}", 
                     help="Weighted F1 - Macro F1 (>0.05 = lớp thiểu số gặp khó khăn)")
        
        if imbalance_diagnosis['is_imbalanced']:
            st.warning(f"""
            Dữ liệu lệch lớp ảnh hưởng mô hình!

{imbalance_diagnosis['message']}  
=> {imbalance_diagnosis['recommendation']}
            """)
        else:
            st.success(imbalance_diagnosis['message'])

        # Nhóm 4: Đánh giá ổn định qua k-Fold Cross Validation
        st.markdown("---")
        st.subheader("5-Fold Cross-Validation")
        
        with st.spinner("Đang thực hiện 5-Fold Cross-Validation..."):
            cv_scores = cross_validate_model(X, y, C=c_param, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        
        # Gọi hàm chẩn đoán từ model.py
        cv_diagnosis = diagnose_cv_stability(cv_mean, cv_std, CV_STABILITY_THRESHOLD)
        
        col_cv1, col_cv2, col_cv3 = st.columns(3)
        with col_cv1:
            st.metric("CV Mean", f"{cv_mean*100:.2f}%", 
                     help="Trung bình accuracy của 5 fold")
        with col_cv2:
            st.metric("CV Std", f"{cv_std*100:.2f}%", 
                     help="Độ lệch chuẩn (nhỏ hơn = ổn định hơn)")
        with col_cv3:
            st.metric("CV Stability", 
                     "Ổn định" if cv_diagnosis['is_stable'] else "Không ổn định",
                     help="Std < 5% => Mô hình ổn định qua các fold")
        
        # Hiển thị score của từng fold
        st.write("Chi tiết từng Fold:")
        fold_df = pd.DataFrame({
            'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
            'Accuracy': [f"{score*100:.2f}%" for score in cv_scores],
            'Giá trị': cv_scores
        })
        st.dataframe(fold_df, use_container_width=True, hide_index=True)
        
        # Hiển thị kết quả chẩn đoán CV
        if cv_diagnosis['is_stable']:
            st.success(f"""
            {cv_diagnosis['message']}
            {cv_diagnosis['recommendation']}
            """)
        else:
            st.warning(f"""
            {cv_diagnosis['message']}
            {cv_diagnosis['recommendation']}
            """)


