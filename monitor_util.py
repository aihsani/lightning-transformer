import psutil
import GPUtil
import time
import pandas as pd
from sqlalchemy import create_engine

# Function to get CPU usage


def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

# Function to get memory usage


def get_memory_usage():
    return psutil.virtual_memory().percent

# Function to get GPU utilization and memory


def get_gpu_stats():
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]  # Assuming you're using the first GPU
        return gpu.load * 100, gpu.memoryUsed, gpu.memoryTotal
    return None, None, None

# Function to get Superset process CPU and memory usage


def get_superset_stats():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'superset' in proc.info['name'] or any('superset' in arg for arg in proc.info['cmdline']):
            return proc.cpu_percent(), proc.memory_percent()
    return None, None


# Database connection (replace with your Superset metadata database details)
engine = create_engine('postgresql://admin:admin@localhost:5432/superset')

# Main monitoring loop


def monitor_resources(interval=5, duration=3600):
    start_time = time.time()
    data = []

    while time.time() - start_time < duration:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cpu_usage = get_cpu_usage()
        mem_usage = get_memory_usage()
        gpu_util, gpu_mem_used, gpu_mem_total = get_gpu_stats()
        superset_cpu, superset_mem = get_superset_stats()

        row = {
            'timestamp': timestamp,
            'cpu_usage': cpu_usage,
            'mem_usage': mem_usage,
            'gpu_util': gpu_util,
            'gpu_mem_used': gpu_mem_used,
            'gpu_mem_total': gpu_mem_total,
            'superset_cpu': superset_cpu,
            'superset_mem': superset_mem
        }
        data.append(row)

        # Print current stats
        print(f"Time: {timestamp}")
        print(f"CPU Usage: {cpu_usage}%")
        print(f"Memory Usage: {mem_usage}%")
        if gpu_util is not None:
            print(f"GPU Utilization: {gpu_util:.2f}%")
            print(f"GPU Memory: {gpu_mem_used}/{gpu_mem_total} MB")
        if superset_cpu is not None:
            print(f"Superset CPU Usage: {superset_cpu}%")
            print(f"Superset Memory Usage: {superset_mem}%")
        print("--------------------")

        # Save to database every 5 minutes
        if len(data) % (5 * 60 / interval) == 0:
            df = pd.DataFrame(data)
            df.to_sql('resource_monitoring', engine, if_exists='append', index=False)
            data = []

        time.sleep(interval)

    # Save any remaining data
    if data:
        df = pd.DataFrame(data)
        df.to_sql('resource_monitoring', engine, if_exists='append', index=False)


if __name__ == "__main__":
    monitor_resources()
