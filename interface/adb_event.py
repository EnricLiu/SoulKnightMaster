from enum import Enum

class EnumBase(Enum):
    def __str__(self):
        return self.name

    def __hash__(self):
        return self.value
    
    @classmethod
    def from_int(cls, value: int):
        for member in cls:
            if member.value == value:
                return member
        
        raise ValueError(f"No member found for value {value}")
    
    @classmethod
    def from_string(cls, value: str):
        for member in cls:
            if member.name.lower() == value.lower():
                return member
        
        raise ValueError(f"No member found for value {value}")
    
    def __eq__(self, other: 'EnumBase') -> bool:
        if isinstance(other, EnumBase):
            return self.value == other.value
        return False

class AdbEvent:
    class Type(EnumBase):
        EV_SYN = 0
        EV_KEY = 1
        EV_ABS = 3
        
    class Event(EnumBase):
        ABS_MT_TRACKING_ID  = 0x0039
        ABS_MT_POSITION_X   = 0x0035
        ABS_MT_POSITION_Y   = 0x0036
        ABS_MT_SLOT         = 0x002f
        KEY_W = 0x0011
        KEY_A = 0x001e
        KEY_S = 0x001f
        KEY_D = 0x0020
        BTN_TOUCH   = 0x014A
        SYN_REPORT  = 0x0000
        
        @property
        def type(self):
            type_valid_name = {
                AdbEvent.Type.EV_ABS: ["ABS"],
                AdbEvent.Type.EV_KEY: ["KEY", "BTN"],
                AdbEvent.Type.EV_SYN: ["SYN"]
            }
            for type, names in type_valid_name.items():
                for name in names:
                    if name in self.name:
                        return type
                    
            raise ValueError(f"Invalid event name {self.name}")
    
    class Value(EnumBase):
        UP      = 0
        DOWN    = 1
        
    def __init__(self, event: Event, value: int|Value):
        self._event = event
        self._value = value

    @staticmethod
    def ev_syn():
        # return AdbEvent(AdbEvent.Event.SYN_REPORT, 0xfffffffc)
        return AdbEvent(AdbEvent.Event.SYN_REPORT, 0x0000)
        
    @property
    def event(self): return self._event
    @property
    def value(self): return self._value
    
    def to_command(self) -> str:
        return f"{self.event.type.value} {self.event.value} {self.value}"
    
    def __eq__(self, other: 'AdbEvent') -> bool:
        if isinstance(other, AdbEvent):
            return self.event == other.event and self.value == other.value
        return False
    
class EventCommand:
    def __init__(self, event_device):
        self._event_device = event_device
        self._commands = []
    
    def append(self, command):
        self._commands.append(command)
        
    def to_command(self):
        ret = ""
        for cmd in self._commands:
            ret += "sendevent "
            ret += self.device
        return 
        
        
        
if __name__ == "__main__":
    event = AdbEvent(AdbEvent.Event.KEY_W, 1)
    print(event.to_command())