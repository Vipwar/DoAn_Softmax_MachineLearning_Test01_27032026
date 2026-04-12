import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

def train_softmax(X_train, y_train, C=1.0):
    """
    Huấn luyện Softmax Regression (Multinomial Logistic Regression).
    Tham số C: regularization strength (C lớn thì ít regularization).
    """
    model = LogisticRegression(
        solver='lbfgs',
        C=C,
        max_iter=2000,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, target_names):
    """Đánh giá trên tập test."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    n_classes = len(target_names)
    all_labels = list(range(n_classes))
    
    report_dict = classification_report(
        y_test, y_pred, 
        labels=all_labels,
        target_names=target_names, 
        output_dict=True,
        zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    
    cm = confusion_matrix(y_test, y_pred, labels=all_labels)
    
    brier_score = calculate_brier_score(y_test, y_proba)
    
    return acc, report_df, cm, brier_score


def calculate_brier_score(y_test, y_proba):
    """
    Tính Brier Score để đánh giá độ tin cậy của xác suất dự đoán.
    """
    n_classes = y_proba.shape[1]
    
    y_test_onehot = np.zeros((len(y_test), n_classes))
    for i, label in enumerate(y_test):
        y_test_onehot[i, label] = 1
    
    brier = np.mean((y_proba - y_test_onehot) ** 2)
    
    return brier


def cross_validate_model(X, y, C=1.0, cv=5):
    """
    Đánh giá mô hình bằng k-Fold Cross-Validation.
    Trả về danh sách accuracy từ các fold để tính Mean ± Std.
    """
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    scores = cross_val_score(
        LogisticRegression(solver='lbfgs', C=C, max_iter=2000, random_state=42),
        X, y, 
        cv=kfold, 
        scoring='accuracy'
    )
    
    return scores


# Chẩn đoán Overfitting/Underfitting dựa trên Gap Accuracy và Macro F1
def diagnose_overfitting_underfitting(train_acc, test_acc, macro_f1, threshold_gap=0.10, threshold_f1=0.65):
    gap = train_acc - test_acc
    
    # Nếu gap lớn và macro_f1 cao → mô hình học thuộc trên lớp đa số
    if gap > threshold_gap and macro_f1 > threshold_f1:
        return {
            'status': 'OVERFITTING',
            'gap': gap,
            'macro_f1': macro_f1,
            'message': f'OVERFITTING: Gap Accuracy = {gap*100:.2f}% (> {threshold_gap*100:.0f}%)\nMacro F1 = {macro_f1:.4f} > {threshold_f1}',
            'recommendation': 'Mô hình học quá tốt trên train'
        }
    
    # Nếu gap lớn nhưng macro_f1 thấp → mô hình học thuộc nhưng bỏ qua lớp thiểu số
    elif gap > threshold_gap and macro_f1 <= threshold_f1:
        return {
            'status': 'OVERFITTING LỆCH LỚP',
            'gap': gap,
            'macro_f1': macro_f1,
            'message': f'OVERFITTING LỆCH LỚP: Gap Accuracy = {gap*100:.2f}% > {threshold_gap*100:.0f}%, \nMacro F1 = {macro_f1:.4f} ( < {threshold_f1})' ,
            'recommendation': 'Mô hình học thuộc và bỏ qua lớp thiểu số '
        }
    
    # Nếu gap nhỏ nhưng macro_f1 thấp → mô hình quá đơn giản
    elif gap <= threshold_gap and macro_f1 <= threshold_f1:
        return {
            'status': 'UNDERFITTING',
            'gap': gap,
            'macro_f1': macro_f1,
            'message': f'UNDERFITTING: Gap Accuracy = {gap*100:.2f}% ≤ {threshold_gap*100:.0f}%, \nMacro F1 = {macro_f1:.4f} (< {threshold_f1})',
            'recommendation': 'Mô hình quá đơn giản, yếu trên tất cả lớp '
        }
    
    # Nếu gap nhỏ và macro_f1 cao → mô hình ổn định, công bằng
    else:
        return {
            'status': 'STABLE',
            'gap': gap,
            'macro_f1': macro_f1,
            'message': f'STABLE: Gap Accuracy = {gap*100:.2f}% ≤ {threshold_gap*100:.0f}%, \nMacro F1 = {macro_f1:.4f} ( ≥ {threshold_f1} )',
            'recommendation': 'Mô hình hoạt động tốt, công bằng với mọi lớp'
        }

# Chẩn đoán ảnh hưởng của dữ liệu mất cân bằng dựa trên Macro F1 vs Weighted F1
def diagnose_imbalanced_data(macro_f1, weighted_f1, threshold=0.05):
    gap = weighted_f1 - macro_f1
    
    # Nếu gap lớn → dữ liệu lệch lớp, mô hình thiên lệch
    if gap > threshold:
        return {
            'is_imbalanced': True,
            'gap': gap,
            'message': f'Dữ liệu lệch lớp: Macro F1 ({macro_f1:.4f}) < Weighted F1 ({weighted_f1:.4f})',
            'recommendation': 'Mô hình hoạt động tốt trên lớp đa số, nhưng kém trên lớp thiểu số'
        }
    # Nếu gap nhỏ → dữ liệu cân bằng
    else:
        return {
            'is_imbalanced': False,
            'gap': gap,
            'message': f'Dữ liệu cân bằng: Macro F1 ~ Weighted F1',
            'recommendation': 'Mô hình công bằng với mọi lớp'
        }

# Kiểm tra tính ổn định của mô hình qua k-Fold Cross Validation
def diagnose_cv_stability(cv_mean, cv_std, threshold=0.05):
    # Nếu độ lệch chuẩn < 5% → mô hình ổn định
    if cv_std < threshold:
        return {
            'is_stable': True,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'message': f'Mô hình ổn định! (Mean = {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%)',
            'recommendation': 'Độ lệch chuẩn < 5% → Mô hình hoạt động nhất quán qua các tập dữ liệu khác nhau'
        }
    # Nếu độ lệch chuẩn >= 5% → mô hình chưa ổn định
    else:
        return {
            'is_stable': False,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'message': f'Mô hình chưa ổn định (Mean = {cv_mean*100:.2f}% +/- {cv_std*100:.2f}%)',
            'recommendation': 'Độ lệch chuẩn >= 5% → Có thể cần điều chỉnh tham số C hoặc thu thập thêm dữ liệu'
        }

