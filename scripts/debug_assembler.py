import time, asyncio
import voicecred.main as vcmain
from fastapi.testclient import TestClient
client = TestClient(vcmain.app)
r = client.post('/sessions/start')
js = r.json(); sid = js['session_id']
print('session', sid)
print('assembler_tasks_has:', sid in vcmain.assembler_tasks)
async def publish():
    await vcmain.bus.publish(sid, 'window_buffer', {'type':'qc','window_id':'w1','qc':{'snr_db':20.0,'speech_ratio':0.9,'voiced_seconds':0.5}})
    await vcmain.bus.publish(sid, 'window_buffer', {'type':'feature_frame','window_id':'w1','feature':{'acoustic_named':{'f0_mean':100.0},'linguistic':{'ttr':0.5}}, 'timestamp_ms':123})
asyncio.get_event_loop().run_until_complete(publish())
wb = vcmain.bus.sessions.get(sid)['window_buffer']
uo = vcmain.bus.sessions.get(sid)['ui_out']
print('window_buffer qsize:', wb.q.qsize())
print('ui_out qsize:',uo.q.qsize())
# wait a bit
for i in range(20):
    print('iter',i, 'wch q', wb.q.qsize(), 'uich q',uo.q.qsize())
    time.sleep(0.1)
print('done')
