from .replay_recorder import SoulKnightReplayRecorder
from pathlib import Path
import json

__REPLAY_RECORDER_PATH: Path = Path(__file__).parent.absolute()

async def record_replay(config, export_path:Path=None, duration_s:int=10):
    export_path = __REPLAY_RECORDER_PATH / "out" if export_path is None else export_path
    if not export_path.is_dir():
        raise ValueError("Invalid export path")
    
    recorder = await SoulKnightReplayRecorder.get_instance(config)
    return await recorder.record(export_path, duration_s, 2)

def default_config():
    return json.load(open(__REPLAY_RECORDER_PATH / "config.json"))
