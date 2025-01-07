import re
import asyncio

from adb_shell.adb_device_async import AdbDeviceAsync
from adb_event import AdbEvent
from position import Position


class Controller:
    def __init__(self, adb: AdbDeviceAsync, event_device="/dev/input/event4", screen_size = Position(1280, 720)):
        self._adb = adb
        self._SCREEN_SIZE = screen_size
        self._event_device = event_device
        self._fingers = {}
        self._last_finger = ""
        self._initialized = False
        
    async def _send_event(self, commands: list[AdbEvent]):
        if commands[-1] != AdbEvent.ev_syn(): commands.append(AdbEvent.ev_syn())
        command = []
        for c in commands:
            command.append(f"sendevent {self._event_device} {c.to_command()}")
        command = " && ".join(command)
        
        print(command)
        
        line = await self._adb.shell(command, read_timeout_s = float("inf"))
        print(line)
         
    def finger_down(self, name: str, pos: Position = None) -> tuple[list[AdbEvent], int]:
        if name in self._fingers: 
            return self.finger_move(name, pos)
        
        print("finger down")
        slot_id = 1
        for _, v in self._fingers.items():
            if slot_id >= v["slot_id"]: slot_id = v["slot_id"] + 1
            
        commands = []
        if len(self._fingers) != 0:
            commands.append(AdbEvent(AdbEvent.Event.ABS_MT_SLOT, slot_id))
        
        commands.append(AdbEvent(AdbEvent.Event.ABS_MT_TRACKING_ID, slot_id))
        # TODO: call self.finger_move()
        commands.append(AdbEvent(AdbEvent.Event.BTN_TOUCH, AdbEvent.Value.DOWN))
        
        # TODO: refaction, move it out
        self._fingers[name] = {
            "slot_id" : slot_id,
            "position": pos,
        }
        self._last_finger = name
        
        pos = self._pos_remap(pos)
        commands.extend([
            AdbEvent(AdbEvent.Event.ABS_MT_POSITION_X, pos.x),
            AdbEvent(AdbEvent.Event.ABS_MT_POSITION_Y, pos.y)
        ])
        
        return commands
    
    def finger_move(self, name: str, pos: Position) -> list[AdbEvent]:
        if name not in self._fingers: 
            raise Exception("finger not found")
        
        commands = [
            AdbEvent(AdbEvent.Event.BTN_TOUCH, AdbEvent.Value.DOWN),
        ]
        
        if name != self._last_finger:
            commands.append(AdbEvent(AdbEvent.Event.ABS_MT_SLOT, self._fingers[name]["slot_id"]))
        
        # TODO: check position
        self._fingers[name]["position"] = pos
        pos = self._pos_remap(pos)
        commands.extend([
            AdbEvent(AdbEvent.Event.ABS_MT_POSITION_X, pos.x),
            AdbEvent(AdbEvent.Event.ABS_MT_POSITION_Y, pos.y)
        ])
        
        return commands
    
    
    def finger_up(self, name: str) -> list[AdbEvent]:
        if name not in self._fingers: 
            raise Exception("finger not found")
        
        commands = []
        
        if name != self._last_finger:
            commands.append(AdbEvent(AdbEvent.Event.ABS_MT_SLOT, self._fingers[name]["slot_id"]))
        
        commands.extend([
            AdbEvent(AdbEvent.Event.ABS_MT_TRACKING_ID, 0xc350),
            AdbEvent(AdbEvent.Event.BTN_TOUCH, AdbEvent.Value.UP),
        ])
        
        return commands
    
    def _pos_remap(self, pos: Position) -> Position:
        return Position(self._SCREEN_SIZE.y - pos.y, pos.x)
    
    
    
class SoulKnightController(Controller):
    _Instances: dict[str, "SoulKnightController"] = None
    
    def get_instance(key, adb: AdbDeviceAsync) -> 'SoulKnightController':

        if key in SoulKnightController._Instances:
            return SoulKnightController._Instances[key]
        
        instance = SoulKnightController(adb)
        SoulKnightController._Instances[key] = instance
        return instance
    
    def __init__(self, adb: AdbDeviceAsync):
        super().__init__(adb)
        self._adb = adb
        
        
        
if __name__ == "__main__":
    import math
    from adb_shell.transport.tcp_transport_async import TcpTransportAsync
    center = Position(260, 575)
    
    async def main():
        addr = "127.0.0.1"
        port = 16384
        adb_protocol = TcpTransportAsync(addr, port)
        adb = AdbDeviceAsync(adb_protocol, default_transport_timeout_s=5)
        await adb.connect(read_timeout_s=1)
        
        controller = Controller(adb)
        await asyncio.sleep(2)
        
        i = 0
        radius = 100
        print("down!!!!!")
        await controller._send_event(controller.finger_down("joystick", center))
        await asyncio.sleep(0.5)
        
        while i < 1000:
            next_pos = center.offset_polar(radius, math.radians(i))
            print("move!!!!!")
            await controller._send_event(controller.finger_move("joystick", next_pos))
            i += 45
        
        await controller._send_event(controller.finger_up("joystick"))
        print("up!!!!!")
    
    asyncio.run(main())