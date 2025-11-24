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


def _main():
    """Run a simple pyannote diarization against `download.wav`.

    Everything is performed inside this function so importing the module
    is safe during pytest collection.
    """
    print('Starting pyannote speaker diarization test')

    start = time.time()
    adapter = PyannoteSpeakerAdapter(model='pyannote/speaker-diarization-community-1', token=hf_api_key)
    print('adapter pipeline loaded:', getattr(adapter, '_pipeline', None) is not None)

    # load audio into memory and pass as waveform dict to avoid torchcodec decoding
    # audio_dict: {'waveform': (channel, time) torch.Tensor or numpy array, 'sample_rate': int}
    audio_dict = None
    try:
        # prefer torchaudio if available (returns waveform torch.Tensor in shape (channels, time))
        try:
            import torchaudio

            waveform, samplerate = torchaudio.load('download.wav')
            audio_dict = {"waveform": waveform, "sample_rate": int(samplerate)}
        except Exception:
            # try soundfile which returns a numpy array
            import soundfile as sf

            data, samplerate = sf.read('download.wav')
            audio_dict = {"waveform": data, "sample_rate": int(samplerate)}
    except Exception:
        # fallback to wave module with numpy
        import wave
        import numpy as np

        with wave.open('download.wav', 'rb') as wf:
            sr = wf.getframerate()
            nchan = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
        arr = np.frombuffer(frames, dtype='<i2')
        if nchan > 1:
            arr = arr.reshape(-1, nchan)
        audio_dict = {"waveform": arr, "sample_rate": int(sr)}

    if audio_dict is None:
        print('Unable to load download.wav; aborting demo run')
        return

    # run the pipeline locally on your computer
    try:
        res = adapter.recognize(audio_dict)
        print('Diarization segments:')
        print(json.dumps(res, indent=2))
    except Exception as exc:
        print('Error running adapter:', str(exc))


if __name__ == '__main__':
    _main()

