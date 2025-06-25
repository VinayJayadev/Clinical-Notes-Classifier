import subprocess
import sys
import os
import time
import webbrowser
import signal
import atexit
import psutil

def is_port_in_use(port):
    """Check if a port is in use"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            return True
    return False

def kill_process_on_port(port):
    """Kill process running on specified port"""
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            # Get connections for this process
            connections = proc.connections()
            for conn in connections:
                if hasattr(conn, 'laddr') and conn.laddr.port == port:
                    os.kill(proc.pid, signal.SIGTERM)
                    time.sleep(1)  # Give it time to terminate
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def run_mlflow_server():
    """Start MLflow server"""
    if is_port_in_use(5000):
        print("MLflow server is already running on port 5000")
        return None
    
    print("Starting MLflow server...")
    mlflow_process = subprocess.Popen(
        ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to start
    time.sleep(5)
    return mlflow_process

def run_model_comparison():
    """Run model comparison script"""
    print("\nRunning model comparison...")
    comparison_process = subprocess.Popen(
        [sys.executable, "src/compare_models.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return comparison_process

def run_streamlit_app():
    """Run Streamlit app"""
    print("\nStarting Streamlit app...")
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "src/classical_ml_app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return streamlit_process

def cleanup(processes):
    """Cleanup function to terminate all processes"""
    print("\nCleaning up processes...")
    for process in processes:
        if process:
            try:
                process.terminate()
            except:
                pass

def main():
    # List to store all processes
    processes = []
    
    # Register cleanup function
    atexit.register(cleanup, processes)
    
    try:
        # Start MLflow server
        mlflow_process = run_mlflow_server()
        processes.append(mlflow_process)
        
        # Wait for MLflow server to be ready
        time.sleep(5)
        
        # Open MLflow UI in browser
        webbrowser.open('http://localhost:5000')
        
        # Run model comparison
        comparison_process = run_model_comparison()
        processes.append(comparison_process)
        
        # Wait for comparison to complete
        comparison_process.wait()
        
        # Start Streamlit app
        streamlit_process = run_streamlit_app()
        processes.append(streamlit_process)
        
        print("\nAll components are running!")
        print("MLflow UI: http://localhost:5000")
        print("Streamlit app will open in your browser")
        print("\nPress Ctrl+C to stop all processes")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Cleanup will be handled by atexit
        pass

if __name__ == "__main__":
    # Kill any existing processes on the ports we need
    kill_process_on_port(5000)  # MLflow
    kill_process_on_port(8501)  # Streamlit
    
    main() 