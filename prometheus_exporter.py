from prometheus_client import start_http_server, Gauge, Counter
import time
import random

METRICS = {
    'accuracy': Gauge('ml_model_accuracy', 'Akurasi model'),
    'precision': Gauge('ml_model_precision', 'Presisi model'),
    'recall': Gauge('ml_model_recall', 'Recall model'),
    'f1': Gauge('ml_model_f1_score', 'F1 Score model'),
    'latency': Gauge('ml_prediction_latency', 'Waktu respon (detik)'),
    'requests': Counter('ml_requests_total', 'Total permintaan prediksi'),
    'errors': Counter('ml_errors_total', 'Total error pada model'),
    'cpu_usage': Gauge('ml_system_cpu_usage', 'Penggunaan CPU'),
    'mem_usage': Gauge('ml_system_memory_usage', 'Penggunaan Memori'),
    'active_users': Gauge('ml_active_users', 'Jumlah pengguna aktif')
}

def update_metrics():
    while True:
        METRICS['accuracy'].set(random.uniform(0.90, 0.98))
        METRICS['requests'].inc()
        METRICS['latency'].set(random.uniform(0.01, 0.05))
        print("Metrik sedang dikirim ke http://localhost:8000 ...")
        time.sleep(5)

if __name__ == '__main__':
    start_http_server(8000) 
    update_metrics()