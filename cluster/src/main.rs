mod node;
mod utils;
mod server;
mod soul_knight;

use dashmap::DashMap;
use std::net::{SocketAddrV4, Ipv4Addr};
use std::sync::LazyLock;
use adb_client::{ADBServer, ADBDeviceExt, ADBServerDevice};
use tokio::sync::watch;
use tokio::task::spawn_blocking;

static ADB_SERVER_DEFAULT_IP: LazyLock<SocketAddrV4> = LazyLock::new(|| {
    SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 5037)
});

static ADB_SERVERS: LazyLock<DashMap<SocketAddrV4, ADBServer>> = LazyLock::new(|| {
    let map = DashMap::new();
    map.insert(*ADB_SERVER_DEFAULT_IP, ADBServer::new(*ADB_SERVER_DEFAULT_IP));
    map
});

pub struct Cluster {
    adb: DashMap<String, AdbConn>,
}

// impl Cluster {
//     pub fn get_device(&self, name: &str) -> Option<AdbConn> {
//         self.node.get(name).map(|x| x.clone())
//     }
// }

impl Cluster {
    pub fn new() -> Self {
        Cluster {
            adb: DashMap::new()
        }
    }
}

pub struct AdbConn {
    conn: ADBServerDevice,
}

#[tokio::main]
async fn main() -> Result<(), soul_knight::NodeError> {
    use soul_knight::*;
    use chrono::Local;

    const FPS: f64 = 1.0;
    let sleep_duration = std::time::Duration::from_millis((1000.0 / FPS) as u64);
    let mut interval = tokio::time::interval(sleep_duration);

    use serde_json;
    let configs: Vec<NodeConfig> = serde_json::from_str(include_str!("../configs/node.json")).unwrap();
    let config = configs.get(0).unwrap();

    let node: Node<16> = Node::new(config.clone(), "127.0.0.1:5037".parse().unwrap());
    let mut watcher: watch::Receiver<NodeWatcherSignal> = node.watch();
    tokio::spawn(async move {
        while watcher.changed().await.is_ok() {
            let signal = watcher.borrow().clone();
            match signal {
                NodeWatcherSignal::Error {node_name, err } => {
                    println!("[Error] Node[{node_name}]: {err}");
                }
                _ => {}
            }
        }
    });

    let _handle = node.schedule().await?;
    let pi = std::f64::consts::PI;
    let mut start = Local::now();
    for i in 0..10 {
        let action = Action::new(i, Some(i as f64 * pi / 4f64), true, true, true);
        interval.tick().await;
        // let action = Action::new(i, Some(i as f64 * pi / 4f64), false, false, false);
        node.tick(NodeTickerSignal::Tick(action)).await.expect("???");
        let now = Local::now();
        println!("------------->tick<------------ [{}ms]", (now - start).num_milliseconds());
        start = now;
    };
    
    node.release().await;
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    Ok(())
}

// #[tokio::main]
// async fn main() -> Result<(), crate::node::error::Error> {
//     let mut ctrl: node::Node<4> = node::Node::default();
//     ctrl.connect().await?;
// 
//     let mut conn = ctrl.get_conn().await?;
//     let mut task = spawn_blocking(move || {
//         conn.shell_command(&["getevent", "/dev/input/event4"],&mut node::EventParser {})?;
//         Ok::<(), crate::node::error::Error>(())
//     });
// 
//     tokio::select! {
//         _ = & mut task => {
//             println!("task done");
//         },
//         _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
//             println!("timeout, trying to abort!");
//             task.abort();
//         }
//     }
// 
// 
//     // conn.shell(&mut cursor, Box::new(EventParser{}))?;
//     Ok(())
// }


#[test]
fn test() -> Result<(), Box<dyn std::error::Error>> {
    // use std::path::Path;
    let mut server = ADB_SERVERS.get_mut(&*ADB_SERVER_DEFAULT_IP).unwrap();
    let res = server.devices_long()?;
    println!("{:?}", res);

    // let mut server = ADBServer::default();
    let mut device = server.get_device_by_name("127.0.0.1:16384")?;
    println!("{:?}", device);
    // let mut device = ADBServerDevice::new(
    //     "127.0.0.1:16384".to_string(), Some(SocketAddrV4::from_str("127.0.0.1:5037").unwrap()));
    // println!("{:?}", device);
    // let res = device.framebuffer(&Path::new("test.png"))?;
    let res = device.framebuffer_bytes()?.len();
    // drop(server);
    println!("{:?}", res);
    Ok(())
}

#[tokio::test]
async fn test_async() -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;
    let mut server = ADB_SERVERS.get_mut(&*ADB_SERVER_DEFAULT_IP).unwrap();
    let res = server.devices_long().unwrap();
    println!("{:?}", res);
    
    let mut device = server.get_device_by_name("127.0.0.1:16384")?;
    tokio::task::spawn_blocking(move || {
        device.framebuffer(&Path::new("test.png")).unwrap();
    }).await.unwrap();
    Ok(())
}
