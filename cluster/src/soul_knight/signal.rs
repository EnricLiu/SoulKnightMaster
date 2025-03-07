use super::Action;
use super::NodeError;

#[derive(Debug, Clone)]
pub enum NodeWatcherSignal {
    Created { node_name: &'static str },
    Ready { node_name: &'static str },
    Halt { node_name: &'static str },
    Dead { node_name: &'static str },
    
    Error { node_name: &'static str, err: String },
}


pub enum NodeTickerSignal {
    Tick(Action),
    Close,
    Kill
}