import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral-small3.1:latest"  # Or any model you've pulled

def ask_ollama(prompt: str) -> str:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False  # Set to True for streaming responses
    }

    response = requests.post(OLLAMA_API_URL, json=payload)
    if response.status_code == 200:
        return response.json().get("response", "").strip()
    else:
        raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

# Simple loop to interact with the model
if __name__ == "__main__":
    print("Simple Ollama AI Agent (type 'exit' to quit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() in {"exit", "quit"}:
            break
        try:
            reply = ask_ollama(user_input)
            print("Ollama:", reply)
        except Exception as e:
            print("Error:", e)
