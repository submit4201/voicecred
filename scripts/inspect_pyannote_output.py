import os, json
import sys, os
# ensure package imports resolve
HERE = os.path.dirname(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
SRC = os.path.join(HERE, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from voicecred.stt import PyannoteSpeakerAdapter
import soundfile as sf
import torch

hf=os.getenv('HF_API_KEY')
print('HF token present:', bool(hf))
adapter = PyannoteSpeakerAdapter(model='pyannote/speaker-diarization-community-1', token=hf)
print('adapter pipeline:', adapter._pipeline)

data, sr = sf.read('download.wav')
if data.ndim == 1:
    wf = torch.from_numpy(data).unsqueeze(0).to(torch.float32)
else:
    wf = torch.from_numpy(data.T).to(torch.float32)

audio = {'waveform': wf, 'sample_rate': int(sr)}
print('calling pipeline...')
out = adapter._pipeline(audio)
print('type(out)=', type(out))
print('dir(out)=', [x for x in dir(out) if not x.startswith('_')])
print('repr(out)=', repr(out)[:1000])

# Try to see if object has attribute 'labels' or 'get_timeline'
if hasattr(out, 'labels'):
    print('out.labels keys:', list(out.labels.keys()))
if hasattr(out, 'get_timeline'):
    try:
        timeline = out.get_timeline()
        print('timeline type:', type(timeline), 'len:', len(timeline))
        for seg in timeline[:5]:
            print('segment:', seg)
    except Exception as e:
        print('get_timeline error', e)

if hasattr(out, 'speaker_diarization'):
    print('speaker_diarization dir:', [x for x in dir(out.speaker_diarization) if not x.startswith('_')])
    try:
        print('first 10 annotations:')
        # iterate through segments with labels if possible
        if hasattr(out.speaker_diarization, 'itersegments'):
            for item in out.speaker_diarization.itersegments(yield_label=True)[:10]:
                print(item)
        elif hasattr(out.speaker_diarization, 'itertracks'):
            for item in out.speaker_diarization.itertracks(yield_label=True):
                print(item)
    except Exception as e:
        print('error iterating speaker_diarization', e)

# Try common fallbacks
if hasattr(out, 'itertracks'):
    print('has itertracks')
    for t in out.itertracks(yield_label=True):
        print('itertrack entry:', t)

print('done')
