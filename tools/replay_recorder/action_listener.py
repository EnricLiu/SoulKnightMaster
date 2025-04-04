import re
import asyncio

from adb_shell.adb_device_async import AdbDeviceAsync

from .utils import Position, RegionBase

class ActionListener:
    def __init__(self, adb: AdbDeviceAsync):
        self._input_states = {
            "KEY": 0b0000,
            "TOUCH": {}
        }
        self._adb = adb
        self._last_touch_position = Position(-1, -1)

        self._is_listening = False

    def get_input_states(self): return self._input_states

    async def listen(self, on_open:callable=None, on_event:callable=None, on_finish:callable=None):
        def touch_state_machine(touch_state, curr_slot, event_name, event_value):
            def ret(state, slot):
                return { "next_state": state, "curr_slot": slot }
            
            # ↓ this is done in the outer fsm
            # always reset when sync recv
            # if event_name == "SYN_REPORT":
            #     return ret("start", curr_slot)

            match touch_state:
                case "start":
                    match event_name:
                        # BTN_TOUCH DOWN -> "tracking"
                        case "BTN_TOUCH":
                            if event_value == "DOWN":
                                if curr_touch_slot is None: return ret("fault", curr_slot)
                                if curr_touch_slot not in self._input_states["TOUCH"]:
                                    self._input_states["TOUCH"][curr_touch_slot] = self._last_touch_position
                                return ret("tracking", curr_slot)
                    
                        # ABS_MT_TRACKING_ID 0xc350|0xc352 -> "wait_btn_up"
                        # ABS_MT_TRACKING_ID others -> "wait_btn_down"
                        case "ABS_MT_TRACKING_ID":
                            event_value = int(event_value, 16)
                            if 0xc400 >= event_value >= 0xc300:
                                return ret("wait_btn_up", curr_slot)
                            else:
                                curr_slot = event_value
                                return ret("wait_btn_down", curr_slot)
                
                case "tracking": # BTN_TOUCH DOWN detected, position update is needed
                    match event_name:
                        case "ABS_MT_POSITION_X":
                            if curr_touch_slot is None: return ret("fault", curr_slot)
                            if curr_touch_slot not in self._input_states["TOUCH"]: return ret("fault", curr_slot)
                            event_value = int(event_value, 16)
                            self._input_states["TOUCH"][curr_touch_slot].set_x(event_value)
                            self._last_touch_position.set_x(event_value)
                            return ret("tracking", curr_slot)
                        case "ABS_MT_POSITION_Y":
                            if curr_touch_slot is None: return ret("fault", curr_slot)
                            if curr_touch_slot not in self._input_states["TOUCH"]: return ret("fault", curr_slot)
                            event_value = int(event_value, 16)
                            self._input_states["TOUCH"][curr_touch_slot].set_y(event_value)
                            self._last_touch_position.set_y(event_value)
                            return ret("tracking", curr_slot)

                case "wait_btn_up":
                    if event_name == "BTN_TOUCH":
                        if event_value == "UP":
                            if curr_touch_slot is None: return ret("fault", curr_slot)
                            if curr_touch_slot not in self._input_states["TOUCH"]: return ret("fault", curr_slot)
                            del self._input_states["TOUCH"][curr_touch_slot]
                            return ret("finish", curr_slot)
                
                case "wait_btn_down":
                    if event_name == "BTN_TOUCH":
                        if event_value == "DOWN":
                            if curr_touch_slot is None: return ret("fault", curr_slot)
                            self._input_states["TOUCH"][curr_touch_slot] = self._last_touch_position
                            return ret("tracking", curr_slot)
                        
            
                case "finish" | "fault":
                    # return ret("fault", curr_slot)
                    pass

            # all unexpected trans-state fallback here
            return ret("fault", curr_slot)

        line_iter = self._adb.streaming_shell(
            "getevent -l",
            read_timeout_s      = float("inf"),
            transport_timeout_s = 999999999
        )
        
        key_bit_map = { "W": 3, "A": 2, "S": 1, "D": 0 }
        self._input_states = {
            "KEY": 0b0000, # W A S D
            "TOUCH": {}
        }
        # "TOUCH": {"slot_id_1": { "X": x, "Y": y }, "slot_id_2": { "X": x, "Y": y }, ...}
        # "TOUCH": {} means curr no touch detected
        
        touch_state = "start"
        main_state = "idle"
        curr_touch_slot = None
        is_open = False
        regexp_splitter = r"^/dev/input/event4: (\S+) +(\S+) +([0-9a-zA-Z]+)" # lines like: "EV_ABS ABS_MT_POSITION_X 0000013f"

        self._is_listening = True
        async for lines in line_iter:
            if not is_open:
                is_open = True
                if on_open is not None: on_open()

            lines = lines.strip()
            for line in lines.split('\n'):
                line = line.strip()
                
                # DEBUG
                # print(line)
                
                result = re.match(regexp_splitter, line)
                if result is None: continue

                event_type = result.group(1)
                if not event_type.startswith("EV_"): continue
                event_type = event_type[3:]
                event_name = result.group(2)
                event_value = result.group(3)


                if event_type == "SYN":
                    if on_event is not None: on_event(self._input_states)
                    main_state = "idle"
                    touch_state = "start"
                    continue
                
                if main_state == "fault": # fault means to drop curr event frame
                    continue

                match event_type:
                    case "KEY":
                        hardware, key_name = event_name.split("_")
                        if hardware == "KEY":
                            if key_name not in key_bit_map:
                                main_state = "fault"
                                continue

                            if event_value == "DOWN":
                                self._input_states["KEY"] |= 1 << key_bit_map[key_name]
                            else:
                                self._input_states["KEY"] &= ~(1 << key_bit_map[key_name])

                        elif hardware == "BTN":
                            if key_name != "TOUCH":
                                main_state = "fault"
                                continue

                            fsm_result = touch_state_machine(touch_state, curr_touch_slot, event_name, event_value)
                            touch_state, curr_touch_slot = fsm_result["next_state"], fsm_result["curr_slot"]

                            if touch_state == "fault":
                                main_state = "fault"
                                continue


                    case "ABS": # must be inside touch event
                        fsm_result = touch_state_machine(touch_state, curr_touch_slot, event_name, event_value)
                        touch_state, curr_touch_slot = fsm_result["next_state"], fsm_result["curr_slot"]
                        
                        if touch_state == "fault":
                            main_state = "fault"
                            continue
        
        if on_finish is not None: on_finish(self._input_states)
        self._is_listening = False

    # async def interrupt(self):
        # if self._listening_pid is None: raise RuntimeError("ActionListener: No process to interrupt")
        # print("shell start")
        # ret = await self._adb.shell(f"kill -INT {self._listening_pid}", transport_timeout_s=2, read_timeout_s=2, timeout_s=2)
        # print("shell ok")
        # if ret.strip() != "":
        #     raise RuntimeError(f"ActionListener: Failed to interrupt pid {self._listening_pid}. {ret}")
        # self._listening_pid = None


async def __test_main():
    import json
    from pathlib import Path
    from adb_shell.adb_device_async import AdbDeviceAsync, TcpTransportAsync

    config = json.load(open("config.json"))
    adb_config = config["ADBConfig"]
    addr    = adb_config.get("adbHost", "127.0.0.1")
    port    = adb_config.get("adbPort", 16384)
    adb_protocol = TcpTransportAsync(addr, port)
    adb = AdbDeviceAsync(adb_protocol, default_transport_timeout_s=5)
    
    result = await adb.connect(read_timeout_s=1)
    if not result or not adb.available:
        raise Exception("adb connect failed")

    listener = ActionListener(adb=adb)
    
    await listener.listen()


if __name__ == "__main__":
    asyncio.run(__test_main())