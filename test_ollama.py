import ollama

try:
    response = ollama.chat(
        model='llama3.2', 
        messages=[{'role': 'user', 'content': 'Is the bridge working?'}]
    )
    print("\n--- RESPONSE ---")
    print(response['message']['content'])
    print("----------------\n")
except Exception as e:
    print(f"Error: {e}")