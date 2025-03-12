mod adb;
mod utils;
mod server;
mod cluster;
pub mod node;

use dashmap::DashMap;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::sync::{Arc, LazyLock};
use adb_client::{ADBDeviceExt, ADBServer};
use tokio::sync::watch;
use crate::node::Node;
use crate::cluster::{Cluster, NodeConfig, NodeError, NodeWatcherSignal, ServerConfig};

static ADB_SERVER_DEFAULT_IP: LazyLock<SocketAddrV4> = LazyLock::new(|| {
    SocketAddrV4::new(Ipv4Addr::new(127, 0, 0, 1), 5037)
});

static ADB_SERVERS: LazyLock<DashMap<SocketAddrV4, ADBServer>> = LazyLock::new(|| {
    let map = DashMap::new();
    map.insert(*ADB_SERVER_DEFAULT_IP, ADBServer::new(*ADB_SERVER_DEFAULT_IP));
    map
});

static SECRET: &str = include_str!("../configs/secret.txt");
static SERVER_CONFIGS: LazyLock<Vec<ServerConfig>> = LazyLock::new(|| {
    let configs = include_str!("../configs/server.json");
    serde_json::from_str(configs).expect("parse server configs error")
});
static CLUSTER: LazyLock<Arc<Cluster>>
    = LazyLock::new(|| Arc::new(Cluster::new(SERVER_CONFIGS.clone())));

#[tokio::main]
async fn main() -> Result<(), Box::<dyn std::error::Error>> {
    use serde_json::from_str;
    
    let node_configs: Vec<NodeConfig> = from_str(include_str!("../configs/node.json"))?;
    // CLUSTER.new_node(node_configs.get(0).expect("no node configs").clone()).await?;
    for node_config in node_configs {
        println!("new node: {:?}", node_config);
        CLUSTER.new_node(node_config).await.expect("new node error");
    }
    
    let app = server::route();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:55555").await?;
    println!("listening on {}", listener.local_addr()?);
    axum::serve(listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    ).await?;
    
    Ok(())
}

#[tokio::test]
async fn test() -> Result<(), NodeError> {
    use serde_json;
    use chrono::Local;
    use cluster::*;

    const FPS: f64 = 5.0;
    let sleep_duration = std::time::Duration::from_millis((1000.0 / FPS) as u64);
    let mut interval = tokio::time::interval(sleep_duration);

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
    for i in 0..100 {
        interval.tick().await;
        // let action = Action::new(i, Some(i as f64 * pi / 4f64), true, true, true);
        let action = SoulKnightAction::new(i, None, true, false, false);
        // let action = Action::new(i, Some(i as f64 * pi / 4f64), false, false, false);
        node.act(NodeSignal::Action(action)).await.expect("???");
        
        
        let now = Local::now();
        println!("------------->tick<------------ [{}ms]", (now - start).num_milliseconds());
        start = now;
    };
    
    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    Ok(())
}

#[tokio::test]
async fn test_deschedule() -> Result<(), NodeError> {
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
    
    println!("start");
    node.schedule().await?;
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    println!("stop");
    node.deschedule().await?;
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    println!("start");
    node.schedule().await?;
    tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    println!("stop");
    node.deschedule().await?;
    println!("ok!");
    
    Ok(())
}

// #[tokio::main]
// async fn main() -> Result<(), crate::adb::error::Error> {
//     let mut ctrl: adb::Node<4> = adb::Node::default();
//     ctrl.connect().await?;
// 
//     let mut conn = ctrl.get_conn().await?;
//     let mut task = spawn_blocking(move || {
//         conn.shell_command(&["getevent", "/dev/input/event4"],&mut adb::EventParser {})?;
//         Ok::<(), crate::adb::error::Error>(())
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


// #[test]
// fn test() -> Result<(), Box<dyn std::error::Error>> {
//     // use std::path::Path;
//     let mut server = ADB_SERVERS.get_mut(&*ADB_SERVER_DEFAULT_IP).unwrap();
//     let res = server.devices_long()?;
//     println!("{:?}", res);
// 
//     // let mut server = ADBServer::default();
//     let mut device = server.get_device_by_name("127.0.0.1:16384")?;
//     println!("{:?}", device);
//     // let mut device = ADBServerDevice::new(
//     //     "127.0.0.1:16384".to_string(), Some(SocketAddrV4::from_str("127.0.0.1:5037").unwrap()));
//     // println!("{:?}", device);
//     // let res = device.framebuffer(&Path::new("test.png"))?;
//     let res = device.framebuffer_bytes()?.len();
//     // drop(server);
//     println!("{:?}", res);
//     Ok(())
// }
// 
// #[tokio::test]
// async fn test_async() -> Result<(), Box<dyn std::error::Error>> {
//     use std::path::Path;
//     let mut server = ADB_SERVERS.get_mut(&*ADB_SERVER_DEFAULT_IP).unwrap();
//     let res = server.devices_long().unwrap();
//     println!("{:?}", res);
//     
//     let mut device = server.get_device_by_name("127.0.0.1:16384")?;
//     tokio::task::spawn_blocking(move || {
//         device.framebuffer(&Path::new("test.png")).unwrap();
//     }).await.unwrap();
//     Ok(())
// }

