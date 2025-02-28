mod adb;
mod utils;

use dashmap::DashMap;
use std::net::{SocketAddrV4, Ipv4Addr};
use std::str::FromStr;
use std::sync::LazyLock;
use adb_client::{ADBServer, ADBDeviceExt, ADBServerDevice};

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
//         self.adb.get(name).map(|x| x.clone())
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

fn main() {
    println!("Hello, world!");
}

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
