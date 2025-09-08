import requests
import tempfile
import winsound

resp_ms = requests.post("http://localhost:8020/infer", json={
    "text": "Name saya Kar Weng, saya ialah seorang lelaki gay?",
    "speaker": "Shafiqah Idayu"
})
with open("ms.wav", "wb") as f: f.write(resp_ms.content)

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    tmp.write(resp_ms.content)
    tmp_path = tmp.name

# Play immediately (blocking until finished)
winsound.PlaySound(tmp_path, winsound.SND_FILENAME)