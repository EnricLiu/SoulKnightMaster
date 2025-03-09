mod position;
pub use position::Position;
use std::sync::atomic::AtomicUsize;
use std::sync::LazyLock;
use chrono::Local;

pub static COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static DEBUG: bool = true;

pub static START: LazyLock<i64> = LazyLock::new(|| {
    Local::now().timestamp_micros()
});

pub fn get_id() -> usize {
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

pub fn log(s: &str) {
    if !DEBUG {
        return;
    }
    println!("[DEBUG] {s}");
}