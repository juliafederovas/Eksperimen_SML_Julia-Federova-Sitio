import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os
 
dagshub.init(repo_owner='juliafederovas', repo_name='Eksperimen_SML_Julia-Federova-Sitio', mlflow=True)

mlflow.autolog()

mlflow.set_experiment("Occupancy_Estimation_Baseline")
base_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(base_dir, "namadataset_preprocessing", "occupancy_processed.csv")
 
if not os.path.exists(path_data):
    raise FileNotFoundError(f"File dataset tidak ditemukan. Pastikan folder 'namadataset_preprocessing' sudah ada di folder 'Membangun_model'!")

df = pd.read_csv(path_data)
X = df.drop(columns=['Room_Occupancy_Count'])
y = df['Room_Occupancy_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
with mlflow.start_run(run_name="RandomForest_Baseline_Julia"):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    # acc = accuracy_score(y_test, y_pred)
    #mlflow.log_metric("accuracy", acc)
    
    print("Model Baseline Selesai dengan Autologging.")