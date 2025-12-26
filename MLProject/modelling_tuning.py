import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

# 1. Hubungkan ke DagsHub 
os.environ['DAGSHUB_USERNAME'] = 'juliafederovas'
os.environ['DAGSHUB_TOKEN'] = os.getenv("MLFLOW_TRACKING_PASSWORD")
dagshub.init(repo_owner='juliafederovas', 
             repo_name='Eksperimen_SML_Julia-Federova-Sitio', 
             mlflow=True)

# 2. Set Eksperimen
mlflow.set_experiment("Occupancy_Estimation_Skilled_Advance")
base_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(base_dir, "namadataset_preprocessing", "occupancy_processed.csv")

# 3. Load Data
if not os.path.exists(path_data):
    raise FileNotFoundError(f"File dataset tidak ditemukan di: {path_data}. Pastikan folder 'namadataset_preprocessing' sudah kamu copy ke dalam folder 'Membangun_model'!")

df = pd.read_csv(path_data)
X = df.drop(columns=['Room_Occupancy_Count'])
y = df['Room_Occupancy_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning & Manual Logging
with mlflow.start_run(run_name="RandomForest_Tuning_Julia"):
    # Grid Search untuk Tuning (Skilled/Advance)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    # MANUAL LOGGING 
    # Log Parameters
    mlflow.log_params(grid_search.best_params_)
    
    # Log Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    
    # LOGGING ARTEFAK TAMBAHAN 
    # Artefak 1: Confusion Matrix
    plt.figure(figsize=(8,6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Julia')
    path_cm = os.path.join(base_dir, "training_confusion_matrix.png")
    plt.savefig(path_cm)
    mlflow.log_artifact(path_cm)
    
    # Artefak 2: Feature Importance Plot
    plt.figure(figsize=(8,6))
    feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.title('Feature Importance - Julia')
    path_fi = os.path.join(base_dir, "feature_importance.png")
    plt.savefig(path_fi)
    mlflow.log_artifact(path_fi)
    
    # Log Model 
    mlflow.sklearn.log_model(best_model, "model")
    
    print(f"Model dilatih dengan akurasi: {acc}")