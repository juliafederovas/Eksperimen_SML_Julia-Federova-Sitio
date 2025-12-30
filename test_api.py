import pandas as pd
import requests
import time

try:
    df = pd.read_csv("Membangun_model/namadataset_preprocessing/occupancy_processed.csv")
    test_data = df.drop(columns=['Room_Occupancy_Count']).head(10) 
    
    print("Mengirim data asli ke API...")
    for i in range(len(test_data)):
        row = test_data.iloc[[i]].to_dict(orient='records')
        response = requests.post("http://127.0.0.1:5001/predict", json=row)
        print(f"Data ke-{i+1}: Status {response.status_code}")
        time.sleep(1)
except Exception as e:
    print(f"Error: {e}")