use std::sync::atomic::AtomicUsize;

pub static COUNTER: AtomicUsize = AtomicUsize::new(0);
pub fn get_id() -> usize {
    COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}