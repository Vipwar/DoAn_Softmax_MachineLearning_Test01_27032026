import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits

def get_dataset(dataset_name):
    """Tải dataset từ sklearn và đuă về cùng định dạng như clean_data()"""
    if dataset_name == "Iris (Hoa Diên Vĩ - 3 Lớp)":
        data = load_iris()
    elif dataset_name == "Wine (Chất lượng rượu - 3 Lớp)":
        data = load_wine()
    else:
        data = load_digits()
        
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    # Đuă target_names thành Python list để consistent với clean_data()
    target_names = list(data.target_names)
    return X, y, target_names