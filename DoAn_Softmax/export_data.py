import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_digits

print("Dang tien hanh trich xuat du lieu...")

# 1. Trích xuất bộ dữ liệu Iris
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['Target'] = iris.target
df_iris.to_csv('Iris_Dataset.csv', index=False, encoding='utf-8')
print("- Da luu Iris_Dataset.csv")

# 2. Trích xuất bộ dữ liệu Wine
wine = load_wine()
df_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
df_wine['Target'] = wine.target
df_wine.to_csv('Wine_Dataset.csv', index=False, encoding='utf-8')
print("- Da luu Wine_Dataset.csv")

# 3. Trích xuất bộ dữ liệu Digits
digits = load_digits()
df_digits = pd.DataFrame(digits.data, columns=digits.feature_names)
df_digits['Target'] = digits.target
df_digits.to_csv('Digits_Dataset.csv', index=False, encoding='utf-8')
print("- Da luu Digits_Dataset.csv")

print("Hoan tat! Ban hay kiem tra thu muc hien tai nhe.")