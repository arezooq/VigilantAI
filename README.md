# VigilantAI

## Prediction Project with RandomForest Model

This project includes various steps for data preprocessing, modeling, and evaluation using the RandomForestClassifier model.

### Steps of the Project

1. **Data Loading**:
   - Data is loaded from a CSV file.

2. **Data Preprocessing**:
   - This includes handling missing values and splitting the data into features and labels.

3. **Data Scaling**:
   - `StandardScaler` is used to scale the data so the model works better with features of different scales.

4. **Data Splitting**:
   - The data is split into training and testing sets.

5. **Model Definition**:
   - A `RandomForestClassifier` model is defined.

6. **Model Evaluation**:
   - The model's accuracy and a detailed classification report are printed using `classification_report`.

7. **Hyperparameter Optimization**:
   - `GridSearchCV` is used to find the best hyperparameters for the model.

8. **Model Saving**:
   - The final model is saved for future predictions.

### Optimization Techniques Used

- Models are optimized using **Cross Validation** and **GridSearchCV** to select the best model.
- Using **StandardScaler** for data scaling helps improve the model's performance and accuracy.



