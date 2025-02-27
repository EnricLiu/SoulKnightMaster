mod adb;

use dashmap::DashMap;
use std::net::{SocketAddrV4, Ipv4Addr};
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
    let mut server = ADB_SERVERS.get_mut(&*ADB_SERVER_DEFAULT_IP).unwrap();
    let res = server.devices_long()?;
    println!("{:?}", res);

    // let mut server = ADBServer::default();
    let mut device = server.get_device_by_name("localhost:16384")?;
    device.shell_command(&["df", "-h"], &mut std::io::stdout());
    // drop(server);
    Ok(())
}
