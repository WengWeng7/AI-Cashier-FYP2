from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import io
from pydub import AudioSegment
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

app = FastAPI(title="ASR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = None

def load_model(model_name="mesolitica/malaysian-whisper-tiny"):
    global pipe
    if pipe is None:
        print("🔄 Loading ASR model...")
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            return_timestamps=False,
            device=0 if torch.cuda.is_available() else -1,
        )
        print("✅ Model loaded.")
    return pipe

@app.on_event("startup")
async def startup_event():
    load_model()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()

    # Convert WebM → WAV (always safe regardless of browser)
    audio_io = io.BytesIO(audio_bytes)
    audio = AudioSegment.from_file(audio_io, format="webm")

    # Resample → 16k mono WAV
    audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)

    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    audio_np, samplerate = sf.read(wav_io)

    if pipe is None:
        load_model()

    result = pipe({"array": audio_np, "sampling_rate": samplerate})
    return JSONResponse({"transcription": result["text"]})

@app.get("/health")
def health_check():
    return {"status": "ok"}
