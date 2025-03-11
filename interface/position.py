import math
from typing import Callable

class Position:
    Epsilon = 1e-6
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def set_x(self, x: float|Callable) -> 'Position':
        if callable(x):
            self.x = x(self)
        else:
            self.x = x
        return self
    
    def set_y(self, y: float|Callable) -> 'Position':
        if callable(y): 
            self.y = y(self)
        else:
            self.y = y
        return self
    
    def to_tuple(self):
        return (self.x, self.y)

    def round_to_tuple(self):
        return (round(self.x), round(self.y))
    
    def swap(self) -> 'Position':
        self.x, self.y = self.y, self.x
        return self

    def angle_to(self, other: 'Position') -> float:
        return math.degrees(math.atan2(other.y - self.y, other.x - self.x))
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((other.x - self.x)**2 + (other.y - self.y)**2)

    def offset_polar(self, radius: float, angle: float) -> 'Position':
        return Position(self.x + radius * math.cos(angle), self.y + radius * math.sin(angle))
    
    def offset_cartesian(self, dx: float, dy: float) -> 'Position':
        return Position(self.x + dx, self.y + dy)
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __eq__(self, other: 'Position') -> bool:
        e = Position.Epsilon
        return (-e) < (self.x - other.x) < e and (-e) < (self.y - other.y) < e

    def __add__(self, other: 'Position'):
        return Position(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Position'):
        return Position(self.x - other.x, self.y - other.y)

if __name__ == "__main__":
    p1 = Position(1, 2)
    print(p1.offset_polar(1, math.pi))