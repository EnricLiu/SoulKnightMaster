
use std::ops::{Add, Sub, Mul, Div};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct Position<N>
    where N:
        Copy +
        Add<Output = N> +
        Sub<Output = N> +
        Mul<Output = N> +
        Div<Output = N> +
        PartialOrd +
        PartialEq +
{
    pub x: N,
    pub y: N,
}

impl<N> Add for Position<N>
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    type Output = Position<N>;

    fn add(self, rhs: Self) -> Self::Output {
        Position {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}
impl<N> Sub for Position<N>
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    type Output = Position<N>;

    fn sub(self, rhs: Self) -> Self::Output {
        Position {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}
impl<N> Mul<Position<N>> for Position<N>
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    type Output = Position<N>;

    fn mul(self, rhs: Position<N>) -> Self::Output {
        Position {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}
impl<N> Div<Position<N>> for Position<N>
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    type Output = Position<N>;

    fn div(self, rhs: Position<N>) -> Self::Output {
        Position {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
        }
    }
}

impl<N> From<(N, N)> for Position<N>
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    fn from(value: (N, N)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}
impl<N> From<&(N, N)> for Position<N>
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    fn from(value: &(N, N)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}
impl<N> From<Position<N>> for (N, N)
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    fn from(value: Position<N>) -> Self {
        (value.x, value.y)
    }
}
impl<N> From<&Position<N>> for (N, N)
where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    fn from(value: &Position<N>) -> Self {
        (value.x, value.y)
    }
}

impl<N> Position<N>
    where N:
    Copy +
    Add<Output = N> +
    Sub<Output = N> +
    Mul<Output = N> +
    Div<Output = N> +
    PartialOrd +
    PartialEq +
{
    pub fn new(x: N, y: N) -> Self {
        Self { x, y }
    }

    pub fn offset_cartesian(mut self, offset: &Self) -> Self {
        self.x = self.x + offset.x;
        self.y = self.y + offset.y;
        self
    }

    pub fn swap(self) -> Self {
        Self {
            x: self.y,
            y: self.x,
        }
    }
}

impl Default for Position<f32> {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl Default for Position<f64> {
    fn default() -> Self {
        Self { x: 0.0, y: 0.0 }
    }
}

impl Default for Position<u8> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<u16> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<u32> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<u64> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<usize> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}

impl Default for Position<i8> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<i16> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<i32> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<i64> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}
impl Default for Position<isize> {
    fn default() -> Self {
        Self { x: 0, y: 0 }
    }
}

impl Position<f64> {
    pub fn round(&self) -> Position<isize> {
        Position {
            x: self.x.round() as isize,
            y: self.y.round() as isize,
        }
    }
    
    pub fn angle(&self) -> f64 {
        self.y.atan2(self.x)
    }

    pub fn angle_to(&self, other: &Position<f64>) -> f64 {
        let pos = *other - *self;
        pos.angle()
    }
    
    pub fn mold(&self) -> f64 {
        self.x.hypot(self.y)
    }
    
    pub fn distance_to(&self, other: &Position<f64>) -> f64 {
        let pos = *other - *self;
        pos.mold()
    }
}

impl Position<u16> {
    pub fn angle(&self) -> f64 {
        (self.y as f64).atan2(self.x as f64)
    }

    pub fn angle_to(&self, other: &Position<u16>) -> f64 {
        let pos = *other - *self;
        pos.angle()
    }

    pub fn mold(&self) -> f64 {
        (self.y as f64).hypot(self.x as f64)
    }

    pub fn distance_to(&self, other: &Position<u16>) -> f64 {
        let pos = *other - *self;
        pos.mold()
    }
}

impl Position<u32> {
    pub fn angle(&self) -> f64 {
        (self.y as f64).atan2(self.x as f64)
    }

    pub fn angle_to(&self, other: &Position<u32>) -> f64 {
        let pos = *other - *self;
        pos.angle()
    }

    pub fn mold(&self) -> f64 {
        (self.y as f64).hypot(self.x as f64)
    }

    pub fn distance_to(&self, other: &Position<u32>) -> f64 {
        let pos = *other - *self;
        pos.mold()
    }
}

impl Position<u64> {
    pub fn angle(&self) -> f64 {
        (self.y as f64).atan2(self.x as f64)
    }

    pub fn angle_to(&self, other: &Position<u64>) -> f64 {
        let pos = *other - *self;
        pos.angle()
    }

    pub fn mold(&self) -> f64 {
        (self.y as f64).hypot(self.x as f64)
    }

    pub fn distance_to(&self, other: &Position<u64>) -> f64 {
        let pos = *other - *self;
        pos.mold()
    }
}

impl Position<usize> {
    pub fn angle(&self) -> f64 {
        (self.y as f64).atan2(self.x as f64)
    }

    pub fn angle_to(&self, other: &Position<usize>) -> f64 {
        let pos = *other - *self;
        pos.angle()
    }

    pub fn mold(&self) -> f64 {
        (self.y as f64).hypot(self.x as f64)
    }

    pub fn distance_to(&self, other: &Position<usize>) -> f64 {
        let pos = *other - *self;
        pos.mold()
    }
}


