mod position;
pub use position::Position;
use std::sync::atomic::AtomicUsize;
use std::sync::LazyLock;
use chrono::{DateTime, Local};
use log::debug;

pub static COUNTER: AtomicUsize = AtomicUsize::new(0);
pub static DEBUG: bool = false;

pub static START: LazyLock<DateTime<Local>> = LazyLock::new(|| {
    Local::now()
});

pub fn get_id() -> usize {
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

pub fn perf_timer() -> Option<DateTime<Local>> {
    if !DEBUG {
        return None;
    }
    Some(Local::now())
}

pub fn perf_log(s: &str, start: Option<DateTime<Local>>) {
    if !DEBUG {
        return;
    }
    if let Some(start) = start {
        let end = Local::now();
        debug!("{s} <-{}ms->", (end - start).num_milliseconds());
    }
}