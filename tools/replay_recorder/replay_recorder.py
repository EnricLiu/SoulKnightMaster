import re
import json
import time
import asyncio
from pathlib import Path
from adb_shell.adb_device_async import AdbDeviceAsync, TcpTransportAsync

from .action_listener import ActionListener
from .screen_recorder import ScreenRecorder
from .utils import Position, RegionBase, CircleRegion, get_region

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
        screen_adb = await instance.get_connected_adb_clone()
        instance.screen_recorder = ScreenRecorder.get_instance(instance_key, screen_adb, adb_workdir, sr_timout)

        ######################  ActionListener Initialize ######################
        al_config = config["ActionListenerConfig"]
        regions = al_config.get("regions")
        action_adb = await instance.get_connected_adb_clone()
        instance.action_listener = SoulKnightActionListener.get_instance(
            instance_key, action_adb,
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

    async def get_connected_adb_clone(self) -> AdbDeviceAsync:
        addr = self.adb_addr()
        port = self.adb_port()
        protocol = TcpTransportAsync(addr, port)
        adb = AdbDeviceAsync(protocol, default_transport_timeout_s=self.adb_timeout)
        result = await adb.connect(read_timeout_s=SoulKnightReplayRecorder._CONN_TIMEOUT)
        if not result or not adb.available:
            raise Exception("adb connect failed")
        return adb

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
        save_path = save_path.joinpath(f"record_{time.strftime('%Y%m%d-%H_%M_%S')}-_out")
        action_path = save_path.joinpath("action.txt")
        screen_path = save_path.joinpath("screen.mp4")
        
        print(f"[INFO] SoulKnightReplayRecorder: ")
        print(f"\t- Action record path: {action_path}")
        print(f"\t- Screen record path: {screen_path}")
        print(f"\t- Screen record duration: {duration_s}s")
        print("\nRecording will start in 5 seconds...")

        record_start_time = None
        action_results = []
        def on_event_cb(*args, **kwargs):
            if record_start_time is None: return
            states = json.dumps(self.action_listener.get_action(), ensure_ascii=False)
            formatted_str = f"{str(time.time_ns() - record_start_time).zfill(15)[:-3]} | {states}\n" # us timestamp
            action_results.append(formatted_str)

        def on_record_start_cb(*args, **kwargs):
            nonlocal record_start_time
            print(f"SoulKnightReplayRecorder: Start Recording!")
            record_start_time = time.time_ns()

        def on_record_finish_cb(*args, **kwargs):
            nonlocal record_start_time
            print(f"Record Finished! Saving...")
            record_start_time = None

        action_task = asyncio.create_task(self.action_listener.listen(on_event=on_event_cb))
        await asyncio.sleep(5)
        save_path.mkdir(parents=True, exist_ok=True)
        screen_task = asyncio.create_task(
            self.screen_recorder.record(screen_path, duration_s, later_timeout, on_record_start_cb, on_record_finish_cb))

        try:
            record_result = await screen_task
        except KeyboardInterrupt:
            print("Recording Interrupted")
            self.screen_recorder._save()
        except Exception as e:
            print(f"Error during recording: {e}")
        finally:
            pass
            # await self.action_listener.interrupt()
            # await listen_task

        with open(action_path, "w", encoding="utf-8") as f:
            f.writelines(iter(action_results))
        
        await asyncio.sleep(1)
        return record_result

    # listen_task, record_task = await record(...)
    

    # @check_initialized
    # async def interrupt(self):
    #     print("SoulKnightReplayRecorder: Interrupt")
    #     results = await asyncio.gather(self.screen_recorder.interrupt(), self.action_listener.interrupt())


    def adb_port(self): return self.adb_protocol._port
    def adb_addr(self): return self.adb_protocol._host

class SoulKnightActionListener(ActionListener):
    _Instances = {}
    # W A S D
    KEY_ANGLE_LOOKUP = {
        0b0000: None,
        0b1000:    0,   # W
        0b0100:  -90,   # A
        0b0010:  180,   # S
        0b0001:   90,   # D
        0b1100:  -45,   # W + A
        0b1010: None,   # W + S
        0b1001:   45,   # W + D
        0b0110: -135,   # A + S
        0b0101: None,   # A + D
        0b0011:  135,   # S + D
        0b1110:  -90,   # W + A + S
        0b0111:  180,   # A + S + D
        0b1011:   90,   # W + S + D
        0b1101:    0,   # W + A + D
        0b1111: None,
    }

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
                "is_moving": False,
                "direction": 0,
            },
            "attack": False,
            "skill":  False,
            "weapon_switching": False,
        }

        movement_inputs = inputs["KEY"] # 0bxxxx
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
                ret["movement"]["is_moving"] = True
                ret["movement"]["direction"] = self._joystick.center().angle_to(pos)

        if not ret["movement"]["is_moving"]:
            angle = SoulKnightActionListener.KEY_ANGLE_LOOKUP[movement_inputs]
            ret["movement"]["is_moving"] = False if angle is None else True
            ret["movement"]["direction"] = 0     if angle is None else angle
        
        return ret

async def print_input_states(listener: ActionListener):
    while True:
        await asyncio.sleep(0.5)
        res = listener.get_input_states()
        print(f"Key: {res['KEY']}", end = "")
        touch = [str(pos) for pos in res["TOUCH"].values()]
        if touch == []: touch = ["None"]
        print(f"\t Touch: {', '.join(touch)}")

async def print_actions(listener: SoulKnightActionListener):
    while True:
        await asyncio.sleep(0.5)
        print(listener.get_action())

async def async_main6(config):
    recorder = await SoulKnightReplayRecorder.get_instance(config)
    action_listener = recorder.action_listener
    
    listen_task = asyncio.create_task(action_listener.listen())
    print_task = asyncio.create_task(print_actions(action_listener))
    
    await listen_task, print_task

async def async_main5(config):
    recorder = await SoulKnightReplayRecorder.get_instance(config)
    ret = await recorder.record(Path("./out/test"), 10, 2)
    print(ret)
    

if __name__ == '__main__':
    config = json.load(open("config.json"))
    policy = asyncio.get_event_loop_policy()
    policy.get_event_loop().set_debug(True)
    asyncio.run(async_main5(config["SoulKnightReplayRecorderConfig"]))



# async def async_main():
#     record_path = Path(f"./out/screen_record_{time.time()}.mp4")
#     recorder = await SoulKnightReplayRecorder.get_instance("127.0.0.1", 16384)
#     result = await recorder.start_screen_record(save_path=record_path, duration_s=5)
#     print(result)

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