import requests

resp_ms = requests.post("http://localhost:8000/infer", json={
    "text": "Selamat datang ke Stesen LRT Kelana Jaya, bagaimana saya boleh membantu anda? Welcome to the LRT Kelana Jaya station, how can I assist you?",
    "speaker": "Husein"
})
with open("ms.wav", "wb") as f: f.write(resp_ms.content)
