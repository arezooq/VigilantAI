import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

# 1. بارگذاری داده‌ها
def load_data(file_path):
    """
    بارگذاری داده‌ها از فایل CSV
    """
    df = pd.read_csv(file_path)
    return df

# 2. پیش‌پردازش داده‌ها
def preprocess_data(df):
    """
    پردازش داده‌ها و تقسیم آن‌ها به ویژگی‌ها و برچسب‌ها
    """
    df = df.dropna()  # حذف مقادیر گم‌شده
    X = df.drop('label', axis=1)  # ویژگی‌ها
    y = df['label']  # برچسب‌ها
    return X, y

# 3. مقیاس‌بندی داده‌ها
def scale_data(X):
    """
    مقیاس‌بندی ویژگی‌ها با استفاده از StandardScaler
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# 4. تقسیم داده‌ها به مجموعه‌های آموزش و تست
def split_data(X, y):
    """
    تقسیم داده‌ها به مجموعه‌های آموزش و تست
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# 5. تعریف مدل‌ها
def create_random_forest_model():
    """
    ایجاد مدل Random Forest
    """
    model = RandomForestClassifier(random_state=42)
    return model

# 6. ارزیابی مدل‌ها
def evaluate_model(model, X_test, y_test):
    """
    ارزیابی مدل
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))

# 7. بهینه‌سازی هایپرپارامترها با استفاده از GridSearchCV
def optimize_model(X_train, y_train):
    """
    بهینه‌سازی مدل با استفاده از GridSearchCV
    """
    param_grid = {
        'n_estimators': [10, 50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f'Best Parameters: {grid_search.best_params_}')
    return grid_search.best_estimator_

# 8. ذخیره مدل برای استفاده مجدد
def save_model(model, filename):
    """
    ذخیره مدل
    """
    joblib.dump(model, filename)

# 9. نقطه ورود اصلی
def main():
    # 1. بارگذاری داده‌ها
    df = load_data('data/attack_reports/Dos_attack_report.csv')

    # 2. پیش‌پردازش داده‌ها
    X, y = preprocess_data(df)

    # 3. مقیاس‌بندی داده‌ها
    X_scaled, scaler = scale_data(X)

    # 4. تقسیم داده‌ها به مجموعه‌های آموزش و تست
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 5. بهینه‌سازی هایپرپارامترها
    optimized_model = optimize_model(X_train, y_train)

    # 6. ارزیابی مدل
    evaluate_model(optimized_model, X_test, y_test)

    # 7. ذخیره مدل
    save_model(optimized_model, 'model/random_forest_model.pkl')

if __name__ == "__main__":
    main()
