#!/usr/bin/env python3
"""
Script to run all API services together
"""

import subprocess
import sys
import os
import time
import signal
import threading

def run_service(script_name, port, service_name):
    """Run a service in a separate process"""
    try:
        print(f"🚀 Starting {service_name} on port {port}...")
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the service to start
        time.sleep(2)
        
        if process.poll() is None:
            print(f"✅ {service_name} started successfully on port {port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start {service_name}")
            print(f"📄 Error: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {service_name}: {e}")
        return None

def check_service_health(port, service_name):
    """Check if a service is healthy"""
    try:
        import requests
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ {service_name} is healthy: {result.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ {service_name} health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ {service_name} is not responding on port {port}")
        return False
    except Exception as e:
        print(f"❌ Error checking {service_name} health: {e}")
        return False

def main():
    """Main function to run all services"""
    
    print("🚀 Starting All IR System APIs")
    print("=" * 50)
    
    # Define services
    services = [
        ("run_preprocessor_api.py", 8001, "Preprocessor API"),
        ("run_tfidf_api.py", 8002, "TF-IDF API"),
        ("run_embedding_api.py", 8003, "Embedding API"),
        ("run_hybrid_api.py", 8004, "Hybrid API"),
        ("run_hybrid_sequential_api.py", 8005, "Hybrid Sequential API")
    ]
    
    processes = []
    
    try:
        # Start all services
        for script, port, name in services:
            process = run_service(script, port, name)
            if process:
                processes.append((process, name, port))
            else:
                print(f"❌ Failed to start {name}. Stopping all services...")
                return
        
        print("\n⏳ Waiting for all services to be ready...")
        time.sleep(5)
        
        # Check health of all services
        print("\n🔍 Checking service health...")
        all_healthy = True
        
        for _, name, port in processes:
            if not check_service_health(port, name):
                all_healthy = False
        
        if all_healthy:
            print("\n🎉 All services are running and healthy!")
            print("\n📡 Service URLs:")
            print("   - Preprocessor API: http://localhost:8001")
            print("   - TF-IDF API: http://localhost:8002")
            print("   - Embedding API: http://localhost:8003")
            print("   - Hybrid API: http://localhost:8004")
            print("   - Hybrid Sequential API: http://localhost:8005")
            print("\n📚 Documentation URLs:")
            print("   - Preprocessor API: http://localhost:8001/docs")
            print("   - TF-IDF API: http://localhost:8002/docs")
            print("   - Embedding API: http://localhost:8003/docs")
            print("   - Hybrid API: http://localhost:8004/docs")
            print("   - Hybrid Sequential API: http://localhost:8005/docs")
            print("\n🔧 Press Ctrl+C to stop all services")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Stopping all services...")
        else:
            print("\n❌ Some services are not healthy. Please check the logs.")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopping all services...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        # Stop all processes
        for process, name, _ in processes:
            try:
                print(f"🛑 Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"❌ Error stopping {name}: {e}")
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 