from fastapi.testclient import TestClient
import time
import sys
sys.path.insert(0, 'src')
import voicecred.main as vcmain
from voicecred.stt import MockSTTAdapter

vcmain.stt_adapter = MockSTTAdapter(override="low confidence test", default_confidence=0.1)
client = TestClient(vcmain.app)

r = client.post('/sessions/start')
print('start resp', r.json())
sid = r.json()['session_id']
token = r.json()['token']

with client.websocket_connect(f"/ws/{sid}?token={token}") as ws:
    print('ack', ws.receive_json())
    for i in range(3):
        ws.send_json({'type':'frame','ts':i,'pcm':[], 'transcript_override':'low confidence test'})
    start = time.time()
    while time.time() - start < 5:
        try:
            msg = ws.receive_json()
            print('MSG:', msg)
        except Exception as e:
            print('recv timeout/err:', e)

print('done')
