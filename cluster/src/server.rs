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
use crate::soul_knight::{Error, FrameBuffer};
use crate::soul_knight::Action;
use crate::utils::log;

static TEST_PNG: LazyLock<FrameBuffer> = LazyLock::new(|| {
    let reader = ImageReader::open("res/test_big.png").unwrap();
    let image = reader.decode().unwrap().to_rgba8();

    FrameBuffer::new("test", 0, image.into_raw())
});

pub fn route() -> Router {
    Router::new()
        .route("/fb", get(get_fb))
        .route("/devices", get(get_all_devices))
        .route("/action", post(post_action))
        .route("/deschedule_all", get(deschedule_all))
        .route("/schedule_all", get(schedule_all))
}

#[derive(Debug, Serialize)]
pub struct ErrResponse<'a> {
    #[serde(skip)]
    status: StatusCode,

    r#type: &'a str,
    msg:  String,
}

impl<'a> ErrResponse<'a> {
    pub fn err(status: StatusCode, r#type: &'a str, msg: String) -> Self {
        ErrResponse {
            status, r#type, msg,
        }
    }
}

impl From<Error> for ErrResponse<'_> {
    fn from(e: Error) -> Self {
        match e {
            Error::NodeError(e) => {
                ErrResponse::err(StatusCode::INTERNAL_SERVER_ERROR, "node error", e.to_string())
            }
            Error::NodeNotFound(_) => {
                ErrResponse::err(StatusCode::NOT_FOUND, "node error", "Node Not Found".to_string())
            }
            Error::ServerErr(e) => {
                ErrResponse::err(StatusCode::INTERNAL_SERVER_ERROR, "server error", e.to_string())
            },
            _ => {
                ErrResponse::err(StatusCode::INTERNAL_SERVER_ERROR, "unknown error", e.to_string())
            }
        }
    }
}

impl IntoResponse for ErrResponse<'_> {
    fn into_response(self) -> Response {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse().unwrap());
        (self.status, headers, Json::from(self)).into_response()
    }
}


# [derive(Debug, Deserialize)]
pub struct GetFbParams {
    node: String,
}

async fn get_fb(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetFbParams>,
) -> Response {
    log(&format!("<GET> framebuffer: {:?}", query));
    let node_name = query.node;

    match CLUSTER.get_fb_by_name(&node_name).await {
        Ok(fb) => {
            let mut headers = HeaderMap::new();
            headers.insert("Content-Type", "application/octet-stream".parse().unwrap());
            let fb = fb.unwrap().2;
            (headers, Bytes::from(fb)).into_response()
        },
        Err(e) => {
            ErrResponse::from(e).into_response()
        }
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

#[derive(Debug, Serialize)]
pub struct DefaultResponse<'a> {
    node: &'a str,
    success: bool,
    error: Option<ErrResponse<'a>>
}

impl<'a> DefaultResponse<'a> {
    pub fn err(node: &'a str, e: ErrResponse<'a>) -> Self {
        DefaultResponse { node, success: false, error: Some(e) }
    }
    
    pub fn ok(node: &'a str) -> Self {
        DefaultResponse { node, success: true, error: None }
    }
    
    pub fn from_result(node: &'a str, result: Result<(), Error>) -> Self {
        match result {
            Ok(_) => DefaultResponse::ok(node),
            Err(e) => DefaultResponse::err(node, e.into()),
        }
    }
}

impl IntoResponse for DefaultResponse<'_> {
    fn into_response(self) -> Response {
        Json::from(self).into_response()
    }
}

async fn schedule(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetScheduleParams>,
) -> Response {
    let node_name = query.node;
    DefaultResponse::from_result(&node_name, CLUSTER.schedule_node(&node_name).await).into_response()
}

async fn deschedule_all(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
) -> Response {
    let all_devices = CLUSTER.all_devices();
    let mut resp = Vec::with_capacity(all_devices.len());
    for status in all_devices {
        let node = status.name;
        resp.push(DefaultResponse::from_result(&node, CLUSTER.deschedule_node(node).await))
    }

    Json::from(resp).into_response()
}

async fn schedule_all(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
) -> Response {
    let all_devices = CLUSTER.all_devices();
    let mut resp = Vec::with_capacity(all_devices.len());
    for status in all_devices {
        let node = status.name;
        resp.push(DefaultResponse::from_result(node, CLUSTER.schedule_node(node).await))
    }
    Json::from(resp).into_response()
}

#[derive(Debug, Deserialize)]
pub struct PostActionParam {
    node:   String,
    action: Action,
}

async fn post_action(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Json(params): Json<Vec<PostActionParam>>,
) -> Response {
    let mut ret: Vec<DefaultResponse> = Vec::with_capacity(params.len());
    
    for param in params.iter() {
        let node_name = &param.node;
        ret.push(DefaultResponse::from_result(
            node_name, CLUSTER.act_by_name(node_name, param.action.clone()).await))
    }
    
    Json::from(ret).into_response()
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
