import requests


def test_api():
    print("Testing /api/generate endpoint...")
    url = "http://127.0.0.1:11434/api/generate"
    try:
        response = requests.post(
            url,
            json={
                "model": "llama3.2:3b",
                "prompt": "hi",
                "stream": False,
                "options": {"temperature": 0.5, "num_ctx": 2048, "num_predict": 150},
            },
            timeout=10,
        )
        print(f"Status: {response.status_code}")
        print(f"Body: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_api()
