import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_softmax(X_train, y_train, learning_rate_inv):
    """
    Huấn luyện mô hình Softmax Regression.
    Tham số C trong sklearn tỉ lệ nghịch với mức độ điều chuẩn (Regularization).
    """
    # multi_class='multinomial' chính là thuật toán Softmax
    model = LogisticRegression(multi_class='multinomial', 
                               solver='lbfgs', 
                               C=learning_rate_inv, 
                               max_iter=2000)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_names):
    """Đánh giá mô hình và trả về các chỉ số"""
    y_pred = model.predict(X_test)
    
    # Tính Accuracy
    acc = accuracy_score(y_test, y_pred)
    
    # Báo cáo chi tiết (Precision, Recall, F1-Score)
    # Chuyển thành dict để dễ hiển thị lên UI
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    # Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    
    return acc, report_df, cm