"""Simple client harness that starts a session via HTTP then opens WS and sends control + fake frames.

Run: C:/workspace/voicecred/.conda/python scripts/client_harness.py
"""
import asyncio
import json
import numpy as np
import websockets
import httpx

BASE = "http://127.0.0.1:8000"

async def run():
    async with httpx.AsyncClient() as client:
        r = await client.post(f"{BASE}/sessions/start")
        js = r.json()
        sid = js["session_id"]
        token = js["token"]
        print("session", sid)

    ws_url = f"ws://127.0.0.1:8000/ws/{sid}?token={token}"

    async with websockets.connect(ws_url) as ws:
        msg = await ws.recv()
        print("server ack:", msg)

        # send reset
        await ws.send(json.dumps({"type":"control","cmd":"reset"}))
        print(await ws.recv())

        # send some fake frames
        for i in range(3):
            pcm = np.random.randint(-30000, 30000, size=1600).astype('int16').tolist()
            await ws.send(json.dumps({"type":"frame","ts": i, "pcm": pcm}))
            print(await ws.recv())

        # finalize
        await ws.send(json.dumps({"type":"control","cmd":"finalize"}))
        print(await ws.recv())

if __name__ == '__main__':
    asyncio.run(run())
