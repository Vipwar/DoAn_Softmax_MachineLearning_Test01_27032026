import pandas as pd
import numpy as np

def generate_student_data(n_samples=1200):
    np.random.seed(42)
    
    # Giờ học (0-20h), Điểm trung bình (0-10), Tỷ lệ chuyên cần (0-100%)
    study_hours = np.random.uniform(2, 18, n_samples)
    attendance = np.random.uniform(40, 100, n_samples)
    previous_score = np.random.uniform(3, 9, n_samples)
    
    # Tạo logic nhãn (Target): 0-> Trượt, 1-> Khá, 2-> Giỏi
    score = (0.4 * study_hours) + (0.3 * attendance / 10) + (0.3 * previous_score)
    target = np.where(score < 7, 0, np.where(score < 11, 1, 2))
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Prev_Score': previous_score,
        'Target': target
    })

    # --- CHÈN SẠN ---
    # 1. Chèn NaN (50 dòng)
    for col in ['Study_Hours', 'Attendance']:
        df.loc[df.sample(25).index, col] = np.nan
        
    # 2. Chèn Outliers 
    df.loc[0, 'Study_Hours'] = 500.0  # Một ngày học 500 tiếng
    df.loc[1, 'Attendance'] = -99.0   # Chuyên cần âm
    
    # 3. Chèn dữ liệu trùng lặp (100 dòng)
    df = pd.concat([df, df.iloc[:100]], ignore_index=True)
    
    # 4. Nhãn bị nhiễu định dạng
    df.loc[2:5, 'Target'] = "2" # Dạng chuỗi

    df.to_csv("Student_Performance_Dirty.csv", index=False)
    print("✅ Đã tạo Student_Performance_Dirty.csv (1300 samples)")

if __name__ == "__main__":
    generate_student_data()