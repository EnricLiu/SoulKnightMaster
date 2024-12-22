import re
import time
import asyncio
from pathlib import Path

from adb_shell.adb_device_async import AdbDeviceAsync, TcpTransportAsync

class ScreenRecorder:
    _Instances = {}

    def get_instance(key, adb: AdbDeviceAsync, adb_workdir:str, default_later_timeout:int=5) -> "ScreenRecorder":
        if key in ScreenRecorder._Instances:
            return ScreenRecorder._Instances[key]
        
        instance = ScreenRecorder(adb, adb_workdir, default_later_timeout)
        ScreenRecorder._Instances[key] = instance
        return instance

    def __init__(self, adb: AdbDeviceAsync, adb_workdir:str, default_later_timeout:int=5):
        self._adb = adb
        self.adb_workdir = adb_workdir
        self.later_timeout = default_later_timeout

        self._recording_pid = None
        self._buf_adb_dir   = None
        self._save_path     = None

    def is_recording(self):
        return self._recording_pid is not None

    async def _save(self):
        if self._save_path is None: raise RuntimeError("ScreenRecorder: save_path is None")
        if self._buf_adb_dir is None: raise RuntimeError("ScreenRecorder: buf_adb_dir is None")
        result = await self._adb.pull(self._buf_adb_dir, self._save_path)
        return result

    async def record(self, save_path: Path, duration_s:int=None, later_timeout:int=None, on_start:callable=None, on_finish:callable=None):
        self._buf_adb_dir = self.adb_workdir + f"/_ScrnRcdr_buf_{time.time()}.mp4"
        self._save_path = save_path

        if not self._adb.available:
            raise Exception("adb not available")
        
        pid_sets = await self._list_screenrecord_pids()
        later_timeout = later_timeout if later_timeout else self.later_timeout
        time_limit_param = f"--time-limit {duration_s}" if duration_s else ""
        line_iter = self._adb.streaming_shell(
            f"screenrecord {self._buf_adb_dir} {time_limit_param} --verbose",
            read_timeout_s      = duration_s + later_timeout,
            transport_timeout_s = duration_s + later_timeout,
            decode = True
        )

        result = {}
        cplt_result = None
        regexp_frame_info = r"^Configuring recorder for (\d+)x(\d+) ([a-zA-Z0-9\.\/]+) at ([a-zA-Z0-9\.\/]+bps)"
        regexp_start_info = r"^Content area is \d+x\d+ at offset x=\d+ y=\d+"
        regexp_video_info = r"^Encoder stopping; recorded (\d+) frames in (\d+) seconds"
        regexp_finish     = r"^Broadcast completed: result=(\d+)"

        is_open = False
        is_start = False
        async for lines in line_iter:
            lines = lines.strip()
            for line in lines.split("\n"):
                line = line.strip()

                if not is_open:
                    pid_sets = (await self._list_screenrecord_pids()) - pid_sets
                    if not pid_sets: raise RuntimeError("ScreenRecorder: No recording process found")
                    self._recording_pid = pid_sets.pop()
                    is_open = True

                if "frame_size" not in result:
                    re_res = re.match(regexp_frame_info, line)
                    if re_res is None: continue
                    result["frame_size"]    =[re_res.group(1), re_res.group(2)]
                    result["encoding"]      = re_res.group(3)
                    result["bitrate"]       = re_res.group(4)
                    continue

                if on_start is not None and not is_start:
                    re_res = re.match(regexp_start_info, line)
                    if re_res is None: continue
                    is_start = True
                    on_start(result)
                    continue

                if "frame_count" not in result:
                    re_res = re.match(regexp_video_info, line)
                    if re_res is None: continue
                    result["frame_count"]   = re_res.group(1)
                    result["duration"]      = re_res.group(2)
                    continue

                if not cplt_result:
                    res = re.match(regexp_finish, line)
                    if res is None: continue
                    if res.group(1).isdigit():
                        cplt_result = int(res.group(1))
                        if cplt_result == 0:
                            if on_finish is not None: on_finish(result)
                        else:
                            raise Exception("adb screen record failed")
                    else:
                        raise Exception("adb finish result is not digit???")

        await self._save()
        # await adb.shell(f"rm {buf_adb_dir}")
        self._recording_pid = None

        return result
    
    async def interrupt(self):
        if self._recording_pid is None:
            raise RuntimeError("ScreenRecorder: No process to interrupt")
        try:
            ret = await self._adb.shell(f"kill -INT {self._recording_pid}", timeout_s=2)
            if ret.strip() != "":
                raise RuntimeError(f"ScreenRecorder: Failed to interrupt pid {self._recording_pid}. {ret}")
        except TimeoutError as e:
            raise RuntimeError(f"ScreenRecorder: Timeout while interrupting pid {self._recording_pid}. {e}")
        finally:
            self._recording_pid = None
            await self._save()

    async def _list_screenrecord_pids(self, timeout:int=2) -> set[int]:
        results = await self._adb.shell("pgrep -l \"screenrecord\"", timeout_s=timeout)
        results = results.strip().split('\n')
        ret = set()
        for result in results:
            result = result.split()
            if len(result) != 2: continue
            ret.add(int(result[0]))
        return ret
