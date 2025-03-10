use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::LazyLock;

use image;
use image::{ImageReader};

use axum::{
    extract::ConnectInfo,
    response::IntoResponse,
    routing::{get, post},
    Json,
    Router
};
use axum::body::Bytes;
use axum::http::{HeaderMap, StatusCode};
use axum::extract::Query;
use axum::response::Response;
use serde::Deserialize;

use crate::soul_knight::{Cluster, FrameBuffer, ServerConfig};
use crate::soul_knight::Action;
use crate::utils::log;

static SECRET: &str = include_str!("../configs/secret.txt");

static SERVER_CONFIGS: LazyLock<Vec<ServerConfig>> = LazyLock::new(|| {
    let configs = include_str!("../configs/server.json");
    serde_json::from_str(configs).expect("parse server configs error")
});

static CLUSTER: LazyLock<Arc<Cluster>>
    = LazyLock::new(|| Arc::new(Cluster::new(SERVER_CONFIGS.clone())));


#[derive(Debug, Clone, Deserialize)]
pub struct WsConnQuery {
    secret: Option<String>
}

impl WsConnQuery {
    pub fn is_valid(&self) -> bool {
        self.secret.as_ref().map(|s| s == SECRET).unwrap_or(false)
    }
}

pub fn route() -> Router {
    Router::new()
        .route("/fb", get(get_fb_test))
        .route("/devices", get(get_all_devices))
        .route("/action", post(post_action))
}


static TEST_PNG: LazyLock<FrameBuffer> = LazyLock::new(|| {
    let reader = ImageReader::open("res/test_big.png").unwrap();
    let image = reader.decode().unwrap().to_rgba8();

    FrameBuffer::new("test", 0, image.into_raw())
});


#[derive(Debug, Deserialize)]
pub struct GetFbParams {
    node: String,
}

// async fn get_fb(
//     ConnectInfo(_addr): ConnectInfo<SocketAddr>,
//     Query(query): Query<Vec<GetFbParams>>,
// ) -> impl IntoResponse {
//     let mut ret = Vec::with_capacity(query.len());
//
//     for param in query {
//         let node_name = param.node;
//         let fb = CLUSTER.get_fb_by_name(&node_name).await;
//         if let Ok(fb) = fb {
//             ret.push(fb);
//         }
//     }
//
//     let mut headers = HeaderMap::new();
//     headers.insert("Content-Type", "application/json".parse().unwrap());
//
//     (headers, Json::from(ret))
//
// }

async fn get_fb_test(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetFbParams>,
) -> Response {
    log(&format!("<GET> framebuffer: {:?}", query));
    let node_name = query.node;
    let mut headers = HeaderMap::new();
    headers.insert("Content-Type", "application/json".parse().unwrap());
    let fb = TEST_PNG.clone().unwrap().2;
    (headers, Bytes::from(fb)).into_response()
}

async fn get_fb(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetFbParams>,
) -> Response {
    log(&format!("<GET> framebuffer: {:?}", query));
    let node_name = query.node;
    let fb = CLUSTER.get_fb_by_name(&node_name).await;
    if let Ok(fb) = fb {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());
        let fb = fb.unwrap().2;
        (headers, Bytes::from(fb)).into_response()
    } else {
        (StatusCode::BAD_REQUEST, "node not found").into_response()
    }
}

async fn get_all_devices(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let devices = CLUSTER.all_devices();
    log(&format!("<GET> all_devices: {:?}", devices));
    Json::from(devices)
}

#[derive(Debug, Deserialize)]
pub struct PostActionParam {
    node:   String,
    action: Action,
}

async fn post_action(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Json(params): Json<Vec<PostActionParam>>,
) -> impl IntoResponse {
    for param in params {
        let node_name = param.node;
        let action = param.action;
        // let _ = CLUSTER.send_action_to_node(&node_name, action).await;
    }
}

#[tokio::test]
async fn test_get_fb() {
    use chrono::Local;

    let req_num = 10;
    let resp = get_fb(ConnectInfo("127.0.0.1:8080".parse().unwrap()), Query(GetFbParams {node: "a".to_string()})).await;
    let start = Local::now();
    for _ in 0..req_num {
        let _res = get_fb(ConnectInfo("127.0.0.1:8080".parse().unwrap()), Query(GetFbParams {node: "a".to_string()})).await;
    }
    let end = Local::now();
    println!("{}ms", (end - start).num_milliseconds() as f64 / req_num as f64);
}
