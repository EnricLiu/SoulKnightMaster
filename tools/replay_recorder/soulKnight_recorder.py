import re
import json
import time
import asyncio
from pathlib import Path
from utils import Position, RegionBase, CircleRegion, get_region

from action_listener import ActionListener
from screen_recorder import ScreenRecorder
from adb_shell.adb_device_async import AdbDeviceAsync, TcpTransportAsync

class SoulKnightReplayRecorder:
    _Instances = {}
    _CONN_TIMEOUT = 5

    @staticmethod
    # async def get_instance(addr:str="127.0.0.1", port:int=16384, timeout:int=5, adb_workdir:str="/sdcard/soul_knight_master"):
    async def get_instance(config: dict):
        adb_config = config["ADBConfig"]
        addr    = adb_config.get("adbHost", "127.0.0.1")
        port    = adb_config.get("adbPort", 16384)
        timeout = adb_config.get("adbTimeout", 5)
        workdir = adb_config.get("adbWorkdir", "/sdcard/soul_knight_master")

        instance_key = f"{addr}:{port}"
        if instance_key in SoulKnightReplayRecorder._Instances:
            return SoulKnightReplayRecorder._Instances[instance_key]

        instance = SoulKnightReplayRecorder(addr, port, timeout)

        ############################ ADB Initialize ############################
        # try to connect
        result = await instance.adb.connect(read_timeout_s=SoulKnightReplayRecorder._CONN_TIMEOUT)
        if not result or not instance.adb.available:
            raise Exception("adb connect failed")
        
        # make workdir
        adb_workdir = workdir if workdir.startswith("/sdcard") else "/sdcard" + workdir
        result = await instance.adb.shell(f"mkdir {adb_workdir}")
        if result.strip():
            raise Exception(f"adb make workdir failed: {result}")
        instance.adb_workdir = adb_workdir

        ######################  ScreenRecorder Initialize ######################
        sr_config = config["ScreenRecorderConfig"]
        sr_timout = sr_config.get("recordTimeout", 5)
        # duration  = sr_config.get("recordDuration", 120)
        instance.screen_recorder = ScreenRecorder.get_instance(instance_key, instance.adb, adb_workdir, sr_timout)

        ######################  ActionListener Initialize ######################
        al_config = config["ActionListenerConfig"]
        regions = al_config.get("regions")
        instance.action_listener = SoulKnightActionListener.get_instance(
            instance_key, instance.adb,
            get_region(**regions[  "joystick"]),
            get_region(**regions[   "btn_atk"]),
            get_region(**regions[ "btn_skill"]),
            get_region(**regions["btn_weapon"])
        )
        
        # save instance & return
        instance.initialized = True
        SoulKnightReplayRecorder._Instances[instance_key] = instance
        return instance

    def __init__(self, adb_addr:str, adb_port:int, adb_timeout:int=5):
        self.initialized = False
        self.adb_protocol = TcpTransportAsync(adb_addr, adb_port)
        self.adb = AdbDeviceAsync(self.adb_protocol, default_transport_timeout_s=adb_timeout)
        self.adb_timeout = adb_timeout

        self.adb_workdir = None

        self.screen_recorder: ScreenRecorder = None
        self.action_listener: SoulKnightActionListener = None

    @staticmethod
    def check_initialized(func):
        async def wrapper(self, *args, **kwargs):
            if not self.initialized:
                raise Exception("SoulKnightReplayRecorder Instance not initialized")
            result = await func(self, *args, **kwargs)
            return result
        return wrapper

    @check_initialized
    async def record(self, save_path: Path, duration_s:int=None, later_timeout:int=None):
        save_path = save_path.joinpath(f"record_{time.strftime('%Y%m%d-%H_%M_%S')}")
        save_path.mkdir(parents=True, exist_ok=True)
        action_path = save_path.joinpath("action.txt")
        screen_path = save_path.joinpath("screen.mp4")
        f_action = open(action_path, "w", encoding="utf-8")

        record_start_time = None
        def on_event_cb(*args, **kwargs):
            nonlocal record_start_time
            if not record_start_time: return
            states = self.action_listener.get_action()
            formatted_str = f"{str(time.time_ns() - record_start_time).zfill(15)} | {states}"
            print(formatted_str)
            f_action.write(formatted_str + "\n")

        def on_record_start_cb(*args, **kwargs):
            nonlocal record_start_time
            record_start_time = time.time_ns()

        def on_record_finish_cb(*args, **kwargs):
            nonlocal record_start_time
            record_start_time = None

        listen_task = asyncio.create_task(self.action_listener.listen(on_event=on_event_cb))
        await asyncio.sleep(5)

        print(f"SoulKnightReplayRecorder: Start Recording!")
        record_task = asyncio.create_task(self.screen_recorder.record(
            screen_path, duration_s, later_timeout, on_record_start_cb, on_record_finish_cb))
        
        try:
            record_result = await record_task
        except KeyboardInterrupt:
            print("Recording Interrupted")
            self.screen_recorder._save()
        finally:
            pass
            # await self.action_listener.interrupt()
            # await listen_task
        
        f_action.close()
        await asyncio.sleep(1)

        return record_result

    # listen_task, record_task = await record(...)
    

    @check_initialized
    async def interrupt(self):
        print("SoulKnightReplayRecorder: Interrupt")
        results = await asyncio.gather(self.screen_recorder.interrupt(), self.action_listener.interrupt())


    def adb_port(self): return self.adb_protocol._port
    def adb_addr(self): return self.adb_protocol._host


class SoulKnightActionListener(ActionListener):
    _Instances = {}
    KEY_ANGLE_MAP = { "W": 0, "A": -90, "S": 180, "D": 0 }

    def get_instance(key, adb: AdbDeviceAsync,
                     joystick_region:   RegionBase, 
                     btn_atk_region:    RegionBase, 
                     btn_skill_region:  RegionBase, 
                     btn_weapon_region: RegionBase,
                    ) -> 'SoulKnightActionListener':
        
        if key in SoulKnightActionListener._Instances:
            return SoulKnightActionListener._Instances[key]
        
        instance = SoulKnightActionListener(adb, joystick_region, btn_atk_region, btn_skill_region, btn_weapon_region)
        SoulKnightActionListener._Instances[key] = instance
        return instance
        
    def __init__(self, adb: AdbDeviceAsync,
                 joystick_region:   RegionBase, 
                 btn_atk_region:    RegionBase, 
                 btn_skill_region:  RegionBase, 
                 btn_weapon_region: RegionBase,
                ):
        
        super().__init__(adb)
        self._joystick   = joystick_region
        self._btn_atk    = btn_atk_region
        self._btn_skill  = btn_skill_region
        self._btn_weapon = btn_weapon_region
    
    def get_action(self):
        curr_inputs = self.get_input_states()
        return self._parse_action_from_inputs(curr_inputs)

    def _parse_action_from_inputs(self, inputs: dict) -> dict:
        ret = {
            "movement": {
                "is_movinng": False,
                "direction":  0,
            },
            "attack": False,
            "skill":  False,
            "weapon_switching": False,
        }

        movement_inputs = inputs["KEY"]
        touch_inputs  = inputs["TOUCH"]

        for _id, pos in touch_inputs.items():
            if self._btn_atk.contains(pos):
                ret["attack"] = True
                continue
            if self._btn_skill.contains(pos):
                ret["skill"] = True
                continue
            if self._btn_weapon.contains(pos):
                ret["weapon_switching"] = True
                continue
            if self._joystick.contains(pos):
                ret["movement"]["is_movinng"] = True
                ret["movement"]["direction"] = self._joystick.center().angle_to(pos)

        if not ret["movement"]["is_movinng"]:
            angle = [SoulKnightActionListener.KEY_ANGLE_MAP[key] for key, value in movement_inputs.items() if value]
            if len(angle) > 0:
                ret["movement"]["is_movinng"] = True
                ret["movement"]["direction"] = sum(angle) / len(angle)
        
        return ret
    
# async def async_main():
#     record_path = Path(f"./out/screen_record_{time.time()}.mp4")
#     recorder = await SoulKnightReplayRecorder.get_instance("127.0.0.1", 16384)
#     result = await recorder.start_screen_record(save_path=record_path, duration_s=5)
#     print(result)

# async def print_input_states(listener: ActionListener):
#     while True:
#         await asyncio.sleep(0.5)
#         res = listener.get_input_states()
#         print(f"Key: {res['KEY']}", end = "")
#         touch = [str(pos) for pos in res["TOUCH"].values()]
#         if touch == []: touch = ["None"]
#         print(f"\t Touch: {', '.join(touch)}")

# async def print_actions(listener: SoulKnightActionListener):
#     while True:
#         await asyncio.sleep(0.5)
#         print(listener.get_action())

# async def async_main2():
#     try:
#         adb_addr = "127.0.0.1"
#         adb_port = 16384
#         recorder = await SoulKnightReplayRecorder.get_instance(adb_addr, adb_port)
#     except Exception as e:
#         print(f"ADB failed to connect {adb_addr}:{adb_port}. {e}")
#         return -1
#     action_listener = ActionListener(recorder.adb)
    
#     # Create tasks for listening and printing input states
#     listen_task = asyncio.create_task(action_listener.listen())
#     print_task = asyncio.create_task(print_input_states(action_listener))
    
#     # Wait for the listen task to complete (it will run indefinitely)
#     await listen_task, print_task
    
# async def async_main3(config):
#     recorder = await SoulKnightReplayRecorder.get_instance(config)
#     action_listener = SoulKnightActionListener(
#         adb = recorder.adb,
#         joystick_region  = CircleRegion(Position( 240, 180).swap(), 269),
#         btn_atk_region   = CircleRegion(Position(1020, 140).swap(),  88),
#         btn_skill_region = CircleRegion(Position(1180, 120).swap(),  59),
#         btn_weapon_region= CircleRegion(Position(1180, 270).swap(),  60)
#     )
    
#     listen_task = asyncio.create_task(action_listener.listen())
#     print_task = asyncio.create_task(print_actions(action_listener))
    
#     await listen_task, print_task

# async def async_main4(config):
#     recorder = await SoulKnightReplayRecorder.get_instance(config)
#     recorder = recorder.screen_recorder
#     res = await recorder.record(Path("./out/screen_record_test.mp4"), 2, 3, lambda x: print("Start!"), lambda x: print("Finish!"))
#     print(res)

async def async_main5(config):
    recorder = await SoulKnightReplayRecorder.get_instance(config)
    ret = await recorder.record(Path("./out/test"), 10, 2)
    print(ret)
    

if __name__ == '__main__':
    config = json.load(open("config.json"))
    asyncio.run(async_main5(config["SoulKnightReplayRecorderConfig"]))

    # action_listener = SoulKnightActionListener(
    #     joystick_region  = CircleRegion(Position( 240, 180), 269),
    #     btn_atk_region   = CircleRegion(Position(1020, 140),  88),
    #     btn_skill_region = CircleRegion(Position(1180, 120),  59),
    #     btn_weapon_region= CircleRegion(Position(1180, 270),  60)
    # )
    # print(action_listener._parse_action_from_inputs({"KEY": {"W": False, "A": False, "S": False, "D": False}, "TOUCH": {"11": Position(1020, 140)}}))

    # self.input_states = {
    #         "KEY": {
    #             "W": False,
    #             "A": False,
    #             "S": False,
    #             "D": False
    #         },
    #         "TOUCH": {}
    #     }
