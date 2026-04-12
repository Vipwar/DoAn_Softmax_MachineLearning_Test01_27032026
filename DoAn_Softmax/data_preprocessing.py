import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def clean_data(df, target_col=None, semantic_order=None):
    # Hàm tiền xử lý dữ liệu bẩn: xóa duplicates, xử lý missing, outliers, mã hóa label
    # semantic_order (optional): List nhãn theo thứ tự ý nghĩa tùy custom (e.g., ["Bad", "Good", "Excellent"])
    
    if target_col is None:
        target_col = df.columns[-1]

    initial_rows = df.shape[0]

    # 1. Xóa các dòng trùng lặp
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Xử lý cột target là các ký tự thì mã hóa thành số
    if df[target_col].dtype == 'object' or df[target_col].dtype.name == 'category':
        unique_labels = sorted([str(x) for x in df[target_col].dropna().unique()])
        
        try:
            numeric_labels = sorted([int(label) for label in unique_labels])
            target_names = [str(x) for x in numeric_labels]
        except ValueError:
            if semantic_order:
                not_in_order = set(unique_labels) - set(semantic_order)
                if not_in_order:
                    raise ValueError(f"Có labels không được định nghĩa trong semantic_order: {not_in_order}")
                    
                target_names = [label for label in semantic_order if label in set(unique_labels)]
            else:
                target_names = unique_labels
        
        label_map = {label: idx for idx, label in enumerate(target_names)}
        df[target_col] = df[target_col].map(label_map)
    else:
        unique_vals = sorted(df[target_col].dropna().unique())
        target_names = [str(int(v)) for v in unique_vals]

    # Chuyển target thành kiểu số
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    
    # XÓA DÒNG có NaN 
    rows_before_drop = len(df)
    df = df.dropna().reset_index(drop=True)
    rows_after_drop = len(df)
    
    if len(df) == 0:
        raise ValueError(f"Không còn dữ liệu! Cột '{target_col}' có thể toàn NaN hoặc không thể convert sang số.")

    # 3. Tách X (feature) và y (target)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    # 4. Xóa các cột non-numeric nếu còn sót
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)

    # 5. Xử lý outliers bằng phương pháp IQR Clipping (chỉ trên cột numeric)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        X[col] = X[col].clip(lower, upper)

    return X.reset_index(drop=True), y.reset_index(drop=True), target_names