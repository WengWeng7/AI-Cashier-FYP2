from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import io
import soundfile as sf
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

app = FastAPI(title="ASR API")

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
    if not file.filename.endswith((".wav", ".flac", ".mp3")):
        raise HTTPException(status_code=400, detail="Invalid audio format. Use WAV, FLAC, or MP3.")

    # Read file into numpy array
    audio_bytes = await file.read()
    audio_np, samplerate = sf.read(io.BytesIO(audio_bytes))

    # Convert stereo → mono
    if len(audio_np.shape) > 1:
        audio_np = np.mean(audio_np, axis=1)

    if pipe is None:
        load_model()

    result = pipe({"array": audio_np, "sampling_rate": samplerate})
    return JSONResponse({"transcription": result["text"]})

@app.get("/health")
def health_check():
    return {"status": "ok"}
