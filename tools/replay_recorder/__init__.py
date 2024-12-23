from .replay_recorder import SoulKnightReplayRecorder
from pathlib import Path

async def record_replay(config, export_path: Path, duration_s: int = 10):
    recorder = await SoulKnightReplayRecorder.get_instance(config)
    return await recorder.record(export_path, duration_s, 2)
