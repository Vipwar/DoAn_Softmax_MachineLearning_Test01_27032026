import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_dataset(dataset_name):
    """Tải bộ dữ liệu tương ứng"""
    if dataset_name == "Iris (Hoa Diên Vĩ - 3 Lớp)":
        data = load_iris()
    elif dataset_name == "Wine (Chất lượng rượu - 3 Lớp)":
        data = load_wine()
    else:
        data = load_digits()
        
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    return X, y, data.target_names

def preprocess_data(X, y, test_size=0.2):
    """Chia tập dữ liệu và Chuẩn hóa (Standardization)"""
    # Chia tập Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Chuẩn hóa dữ liệu để thuật toán Softmax hội tụ nhanh hơn
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test