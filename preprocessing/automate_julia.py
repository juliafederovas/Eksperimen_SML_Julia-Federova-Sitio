import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan di: {path}")
    return pd.read_csv(path)

def preprocess_data(df):
    target_col = 'Room_Occupancy_Count'
    cols_to_drop = ['Date', 'Time']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(X_scaled, columns=X.columns)
    df_processed[target_col] = y.values
    return df_processed

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "../namadataset_raw/Occupancy_Estimation.csv")
    output_path = os.path.join(base_dir, "namadataset_preprocessing/occupancy_processed.csv")

    df_raw = load_data(input_path)
    df_processed = preprocess_data(df_raw)
    save_data(df_processed, output_path)
    print(f"Preprocessing selesai. File tersimpan di: {output_path}")