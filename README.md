# Health Risk Prediction System

Minor project. Web app that predicts the risk of diabetes and heart disease from basic clinical values using ML models. Backend is Flask and the frontend is plain HTML/CSS/JS.

## Datasets

- **Pima Indians Diabetes** (`dataset/diabetes.csv`): 768 rows, 8 features, target `Outcome`
- **Heart Failure Prediction** (`dataset/heart.csv`): 918 rows, 11 features, target `HeartDisease`

## Folder structure

```
dataset/          csv files
notebooks/        eda.ipynb
models/           saved models, scalers, metrics (made by the train scripts)
plots/            charts (made by the train scripts)
static/           css, js, and a copy of the plots for the website
templates/        index.html
train_diabetes_model.py
train_heart_model.py
app.py
```

## How to run

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_diabetes_model.py
python train_heart_model.py
python app.py
```

Then open http://127.0.0.1:5000

## What I did

- Columns like Glucose, Insulin, BMI (diabetes) and Cholesterol, RestingBP (heart) had 0 values, which isn't possible medically. Treated them as missing and filled them with the median of the training data (after the split, so there is no leakage).
- Text columns in the heart data (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope) were one-hot encoded with `get_dummies`.
- 80/20 stratified train/test split, StandardScaler on features.
- Trained 6 models: Logistic Regression, KNN, SVM, Decision Tree, Random Forest, Gradient Boosting.
- Compared them on accuracy, precision, recall, F1, ROC-AUC and 5-fold CV accuracy. The best one is picked by CV accuracy because the test set is small and a single split can be lucky.
- The best model, scaler and column order are saved with joblib, and `app.py` loads them at startup.

## Results

**Diabetes**

| Model | CV Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression (best) | 78.7% | 70.8% | 60.0% | 50.0% | 54.5% | 0.813 |
| SVM | 77.9% | 74.0% | 65.2% | 55.6% | 60.0% | 0.796 |
| Random Forest | 77.4% | 73.4% | 65.1% | 51.9% | 57.7% | 0.810 |
| KNN | 75.6% | 73.4% | 63.3% | 57.4% | 60.2% | 0.788 |
| Gradient Boosting | 75.2% | 74.7% | 65.3% | 59.3% | 62.1% | 0.826 |
| Decision Tree | 73.0% | 76.0% | 63.9% | 72.2% | 67.8% | 0.762 |

Glucose, BMI, Age and Pregnancies matter the most. Decision Tree has the best test accuracy but the worst CV accuracy, so it just got lucky on that split.

**Heart disease**

| Model | CV Acc | Test Acc | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| SVM (best) | 85.6% | 85.3% | 85.0% | 89.2% | 87.1% | 0.941 |
| Gradient Boosting | 85.4% | 86.4% | 88.1% | 87.3% | 87.7% | 0.918 |
| Random Forest | 85.3% | 84.2% | 85.4% | 86.3% | 85.9% | 0.924 |
| KNN | 85.2% | 87.5% | 88.3% | 89.2% | 88.8% | 0.942 |
| Logistic Regression | 84.7% | 89.1% | 89.4% | 91.2% | 90.3% | 0.933 |
| Decision Tree | 81.3% | 72.8% | 75.5% | 75.5% | 75.5% | 0.771 |

The top 5 are all within about 1% of each other.

## Website

- Diabetes and heart disease forms. The result shows as a gauge with Low (<30%), Moderate (30-60%) or High (60%+) risk.
- Combined page: the average of both predictions.
- Model insights page: the comparison table and charts for both diseases.

API routes: `POST /api/predict/diabetes`, `POST /api/predict/heart`, `GET /api/model-info`

## Note

This is an academic project, not a medical tool.

## Future scope

- Hyperparameter tuning with GridSearchCV
- SHAP for explaining individual predictions
- Bigger dataset
- Deploy online
