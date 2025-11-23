import json, time, os, sys
import dotenv
dotenv.load_dotenv(".env")
# check HF_API_KEY loaded
hf_api_key = os.getenv("HF_API_KEY")
print("HF_API_KEY loaded:", hf_api_key is not None)
# ensure package imports resolve (same as scripts/test_real_audio.py)
HERE = os.path.dirname(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SRC = os.path.join(HERE, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from voicecred.stt import PyannoteSpeakerAdapter

print('Starting pyannote speaker diarization test')
# load model and run diarization on download.wav

start = time.time()
adapter = PyannoteSpeakerAdapter(model='pyannote/speaker-diarization-community-1', token=hf_api_key)
print('adapter pipeline loaded:', adapter._pipeline is not None)

# load audio into memory and pass as waveform dict to avoid torchcodec decoding
#  {'waveform': (channel, time) torch.Tensor, 'sample_rate': int}

try:
    # prefer torchaudio if available (returns waveform torch.Tensor in shape (channels, time))
    try:
        import torchaudio
        waveform, samplerate = torchaudio.load('download.wav')
        audio_dict = {"waveform": waveform, "sample_rate": int(samplerate)}
    except Exception:
        import soundfile as sf
        data, samplerate = sf.read('download.wav')
    channel = 1 if len(data.shape) == 1 else data.shape[1]
    time = data.shape[0]
    # pass numpy array into adapter so it can normalize/transpose to a (channels, time) torch.Tensor
    audio_dict = {"waveform": data, "sample_rate": int(samplerate)}
except Exception:
    # fallback to wave module with numpy
    import wave, array
    with wave.open('download.wav', 'rb') as wf:
        sr = wf.getframerate()
        nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
    import numpy as np
    arr = np.frombuffer(frames, dtype='<i2')
    if nchan > 1:
        arr = arr.reshape(-1, nchan)
    data = arr
    samplerate = sr
    audio_dict = {"waveform": data, "sample_rate": int(samplerate)}
# run the pipeline locally on your computer
# call the pipeline directly to inspect returned object for debug
res = adapter.recognize(audio_dict)
print('Diarization segments:')
print(json.dumps(res, indent=2))

