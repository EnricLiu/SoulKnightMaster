import math
from abc import ABC, abstractmethod
from .position import Position

class RegionBase(ABC):
    @abstractmethod
    def contains(self, pos: Position) -> bool: pass
    
    @abstractmethod
    def area(self) -> float: pass

    @abstractmethod
    def center(self) -> Position: pass


class CircleRegion(RegionBase):
    def __init__(self, center: Position | dict, radius: float):
        super().__init__()
        if isinstance(center, dict): center = Position(**center)
        self._center = center
        self._radius = radius
        self._area = math.pi * radius * radius

    def area(self) -> float:
        return self._area
    
    def contains(self, pos: Position) -> bool:
        if pos is None: return False
        if pos.x < self._center.x - self._radius or pos.x > self._center.x + self._radius: return False
        if pos.y < self._center.y - self._radius or pos.y > self._center.y + self._radius: return False
        return (pos.x - self._center.x) ** 2 + (pos.y - self._center.y) ** 2 <= self._radius ** 2
    
    def center(self) -> Position:
        return self._center
    
    def __str__(self) -> str:
        return f"CircleRegion(center={str(self._center)}, radius={self._radius})"


class RectRegion(RegionBase):
    def __init__(self, top_left: Position, bottom_right: Position, width_height: Position):
        super().__init__()
        self.top_left       = top_left     if top_left is not None else bottom_right - width_height
        self.bottom_right   = bottom_right if bottom_right is not None else top_left + width_height
        self.width_height   = width_height if width_height is not None else bottom_right - top_left
        self._area = width_height.x * width_height.y

    def area(self) -> float:
        return self._area

    def contains(self, pos: Position) -> bool:
        if pos is None: return False
        return self.top_left.x < pos.x < self.bottom_right.x and self.top_left.y < pos.y < self.bottom_right.y
    
    def center(self) -> Position:
        return Position(self.top_left.x + self.width_height.x / 2, self.top_left.y + self.width_height.y / 2)
    
    def __str__(self) -> str:
        return f"RectRegion(top_left={str(self.top_left)}, bottom_right={str(self.bottom_right)}, width_height={str(self.width_height)})"
    

if __name__ == "__main__":
    # test
    joystick_region  = CircleRegion(Position( 240, 540), 269)
    test_position = Position( 240, 540)

    print(joystick_region.contains(test_position))
