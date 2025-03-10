use std::net::SocketAddr;
use std::sync::LazyLock;

use image;
use image::ImageReader;

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
use serde::{Deserialize, Serialize};
use serde_json::json;
use crate::{CLUSTER, SECRET};
use crate::soul_knight::FrameBuffer;
use crate::soul_knight::Action;
use crate::utils::log;


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
        .route("/fb", get(get_fb))
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

async fn get_fb(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetFbParams>,
) -> Response {
    log(&format!("<GET> framebuffer: {:?}", query));
    let node_name = query.node;
    let fb = CLUSTER.get_fb_by_name(&node_name).await;
    if let Ok(fb) = fb {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/octet-stream".parse().unwrap());
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

pub struct GetScheduleParams {
    node: String,
}

// async fn get_schedule(
//     ConnectInfo(_addr): ConnectInfo<SocketAddr>,
//     Query(query): Query<GetScheduleParams>,
// ) -> impl IntoResponse {
//     let node_name = query.node;
//     let schedule = CLUSTER.schedule_node(&node_name).await;
//     if let Ok(schedule) = schedule {
//         let mut headers = HeaderMap::new();
//         headers.insert("Content-Type", "application/json".parse().unwrap());
//         (headers, Json::from(json!(r#"{"node": {}"#))).into_response()
//     } else {
//         (StatusCode::BAD_REQUEST, "node not found").into_response()
//     }
// }

#[derive(Debug, Deserialize)]
pub struct PostActionParam {
    node:   String,
    action: Action,
}

#[derive(Debug, Serialize)]
pub struct PostActionResponse {
    success:    bool,
    node:       String,
    msg:        Option<String>,
}

impl PostActionResponse {
    pub fn ok(node: String) -> Self {
        PostActionResponse {
            success: true,
            msg: None,
            node,
        }
    }
    
    pub fn err(node: String, msg: String) -> Self {
        PostActionResponse {
            success: false,
            msg: Some(msg),
            node,
        }
    }
}

async fn post_action(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Json(params): Json<Vec<PostActionParam>>,
) -> impl IntoResponse {
    let mut ret: Vec<PostActionResponse> = Vec::with_capacity(params.len());
    
    for param in params {
        let node_name = param.node;
        let action = param.action;
        if let Err(e) = CLUSTER.act_by_name(&node_name, action).await {
            ret.push(PostActionResponse::err(node_name, e.to_string()));
        } else {
            ret.push(PostActionResponse::ok(node_name));
        }
    }
    
    Json::from(ret)
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
