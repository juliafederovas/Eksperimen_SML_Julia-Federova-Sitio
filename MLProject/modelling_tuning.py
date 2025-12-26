import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

mlflow.set_experiment("Occupancy_Final_Fix")

base_dir = os.path.dirname(os.path.abspath(__file__))
path_data = os.path.join(base_dir, "namadataset_preprocessing", "occupancy_processed.csv")

df = pd.read_csv(path_data)
X = df.drop(columns=['Room_Occupancy_Count'])
y = df['Room_Occupancy_Count']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    mlflow.sklearn.log_model(model, "model")
    print("Training Berhasil!")