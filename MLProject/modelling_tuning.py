import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

mlflow.set_experiment("Occupancy_CI")

base_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(base_dir, "namadataset_preprocessing", "occupancy_processed.csv")

# Load Data
df = pd.read_csv(path_data)
X = df.drop(columns=['Room_Occupancy_Count'])
y = df['Room_Occupancy_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training dengan MLflow lokal
with mlflow.start_run(run_name="Final_CI_Run"):
    param_grid = {'n_estimators': [50], 'max_depth': [10]} 
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=2)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    acc = accuracy_score(y_test, best_model.predict(X_test))
    
    mlflow.log_metric("accuracy", acc)
    
    # Log model ke folder 'model'
    mlflow.sklearn.log_model(best_model, "model")
    print(f"Akurasi: {acc}")