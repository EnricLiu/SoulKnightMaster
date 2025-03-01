mod position;
pub use position::Position;
use std::sync::atomic::AtomicUsize;

pub static COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static DEBUG: bool = false;

pub fn get_id() -> usize {
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

pub fn log(s: &str) {
    if !DEBUG {
        return;
    }
    println!("{s}");
}