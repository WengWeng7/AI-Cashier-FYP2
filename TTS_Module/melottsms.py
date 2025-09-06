from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse
from huggingface_hub import hf_hub_download
from melo.api import TTS
import uuid
import os

# Load model once at startup
ckpt_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='model.pth')
config_path = hf_hub_download(repo_id='mesolitica/MeloTTS-MS', filename='config.json')
model = TTS(language='MS', config_path=config_path, ckpt_path=ckpt_path)

speaker_id = {
    'Husein': 0,
    'Shafiqah Idayu': 1,
    'Anwar Ibrahim': 2,
}

class TTSRequest(BaseModel):
    text: str
    speaker: str = "Husein"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],  # <-- allow OPTIONS, POST, GET
    allow_headers=["*"],
)

@app.post("/infer")
async def infer(request: TTSRequest):
    spk = speaker_id.get(request.speaker, 0)
    filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join("/tmp", filename)

    model.tts_to_file(
        request.text,
        spk,
        output_path,
        split=True,
        sdp_ratio=0,
        noise_scale=0.667
    )

    return FileResponse(output_path, media_type="audio/wav", filename=filename)
