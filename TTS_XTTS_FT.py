import torch
import time
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torchaudio

# Path to your trained model
MODEL_PATH = "C:/Users/User/Downloads/trained_model/content/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT-August-10-2025_03+40PM-dbf1a08a"

config = XttsConfig()
config.load_json(f"{MODEL_PATH}/config.json")

# Set the tokenizer path in model arguments
config.model_args = config.model_args or {}
config.model_args['tokenizer_file'] = f"{MODEL_PATH}/vocab.json"

model = Xtts.init_from_config(config)
model.load_checkpoint(config, 
                     checkpoint_path=f"{MODEL_PATH}/best_model.pth",
                     eval=True,
                     strict=False)  # Allow loading with missing/unexpected keys
model.cuda()

import os
import scipy.io.wavfile as wavfile

# Create output directory if it doesn't exist
os.makedirs("tts_output", exist_ok=True)

outputs = model.synthesize(
    "Welcome! I am the MRT/LRT kiosk assistant, how may I help you.",
    config,
    speaker_wav="tts_clone/audio.wav",
    gpt_cond_len=3,
    language="en",
)

# Save the audio
output_path = "tts_output/output.wav"
wavfile.write(output_path, rate=24000, data=outputs["wav"])
print(f"Audio saved to: {output_path}")
