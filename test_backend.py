
import requests
import time
import sys
import json

BASE_URL = "http://127.0.0.1:8000"

def check_health():
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ Backend is healthy!")
            return True
        else:
            print(f"✗ Backend returned status {response.status_code}: {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("Waiting for backend...")
        return False

def test_query():
    print("\nTesting Query Endpoint...")
    payload = {
        "message": "Hello, is the backend working?",
        "session_id": "test-session"
    }
    try:
        response = requests.post(f"{BASE_URL}/query", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✓ Query successful!")
            print(f"Response: {json.dumps(data, indent=2)}")
            return True
        else:
            print(f"✗ Query failed with status {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Exception during query: {e}")
        return False

def main():
    print("Checking backend connectivity...")
    
    # Wait for backend to come up (max 30 seconds)
    for _ in range(10):
        if check_health():
            break
        time.sleep(3)
    else:
        print("✗ Backend failed to start or is not reachable.")
        sys.exit(1)

    # Test query
    if test_query():
        print("\n✓ Backend and Frontend (simulated) connectivity verified.")
    else:
        print("\n✗ Connectivity check failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
