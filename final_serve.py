import os
import subprocess

os.environ["MLFLOW_SERVER_MODEL_PATH"] = "serving_model"
os.environ["__pyfunc_model_path__"] = "serving_model"

print("Memaksa MLflow membaca folder: serving_model...")

try:
    # Memanggil uvicorn secara langsung
    subprocess.run(["python", "-m", "uvicorn", "mlflow.pyfunc.scoring_server.app:app", "--host", "127.0.0.1", "--port", "5001"])
except Exception as e:
    print(f"Terjadi kesalahan: {e}")