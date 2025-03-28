mod model;
mod response;

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
use axum::http::HeaderMap;
use axum::extract::Query;
use axum::response::{Html, Response};
use log::debug;
use serde::{Deserialize, Serialize};

use crate::cluster::{FrameBuffer, NodeConfig};
use crate::CLUSTER;
use model::{
    GetFbParams, PostActionParam, ScheduleParams,
    // DropNodeParam, NewNodeParam,
};
use response::{DefaultResponse, ErrResponse};
use crate::server::model::GetStatusParams;

static TEST_PNG: LazyLock<FrameBuffer> = LazyLock::new(|| {
    let reader = ImageReader::open("res/test_big.png").unwrap();
    let image = reader.decode().unwrap().to_rgba8();

    FrameBuffer::new("test", 0, image.into_raw())
});

pub fn route() -> Router {
    Router::new()
        .route("/", get(get_index))
        
        .route("/fb", get(get_fb))
        .route("/action", post(post_action))
        
        .route("/status", get(get_status))
        .route("/all_node", get(get_all_nodes))
        // .route("/new_node", post(post_new_node))
        // .route("/drop_node", post(post_drop_node))
        
        .route("/schedule", get(schedule))
        .route("/schedule_all", get(schedule_all))
        .route("/deschedule", get(deschedule))
        .route("/deschedule_all", get(deschedule_all))
}

async fn get_index() -> Response {
    let content = tokio::fs::read_to_string("res/index.html").await;
    Html::from(content.unwrap()).into_response()
}

async fn get_fb(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetFbParams>,
) -> Response {
    debug!("<GET> framebuffer: {:?}", query);
    let node_name = query.node();

    match CLUSTER.get_fb_by_name(node_name).await {
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

async fn post_action(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Json(params): Json<Vec<PostActionParam>>,
) -> Response {
    let mut ret: Vec<DefaultResponse> = Vec::with_capacity(params.len());

    for param in params.iter() {
        let node_name = param.node();
        ret.push(DefaultResponse::from_result(
            node_name, CLUSTER.act_by_name(node_name, param.action()).await.map(|_| None)))
    }

    Json::from(ret).into_response()
}

// async fn post_new_node(
//     ConnectInfo(_addr): ConnectInfo<SocketAddr>,
//     Json(params): Json<NewNodeParam>,
// ) -> Response {
//     let node_config: NodeConfig = params.into();
//     DefaultResponse::from_result(
//         &node_config.name(), CLUSTER.new_node(node_config).await.map(|_| None)).into_response()
// }
// 
// async fn post_drop_node(
//     ConnectInfo(_addr): ConnectInfo<SocketAddr>,
//     Json(params): Json<DropNodeParam>,
// ) -> Response {
//     DefaultResponse::from_result(
//         &params.node(), CLUSTER.drop_node(&params.node()).await.map(|_| None)).into_response()
// }

async fn get_status(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<GetStatusParams>,
) -> Response {
    let node = query.node();
    match CLUSTER.get_status_by_name(node) {
        Ok(status) => { Json::from(status).into_response() },
        Err(e) => { ErrResponse::from(e).into_response() }
    }
}

async fn get_all_nodes(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
) -> impl IntoResponse {
    let devices = CLUSTER.all_devices();
    debug!("<GET> all_devices: {:?}", devices);
    Json::from(devices)
}

async fn schedule(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<ScheduleParams>,
) -> Response {
    let node_name = query.node();
    DefaultResponse::from_result(
        &node_name, CLUSTER.schedule_node(&node_name).await.map(|_| None)).into_response()
}

async fn deschedule(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
    Query(query): Query<ScheduleParams>,
) -> Response {
    let node_name = query.node();
    DefaultResponse::from_result(
        &node_name, CLUSTER.deschedule_node(&node_name).await.map(|_| None)).into_response()
}

async fn schedule_all(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
) -> Response {
    let all_devices = CLUSTER.all_devices();
    let mut resp = Vec::with_capacity(all_devices.len());
    for status in all_devices {
        let node = status.name;
        resp.push(DefaultResponse::from_result(
            node, CLUSTER.schedule_node(node).await.map(|_| None)))
    }
    Json::from(resp).into_response()
}

async fn deschedule_all(
    ConnectInfo(_addr): ConnectInfo<SocketAddr>,
) -> Response {
    let all_devices = CLUSTER.all_devices();
    let mut resp = Vec::with_capacity(all_devices.len());
    for status in all_devices {
        let node = status.name;
        resp.push(DefaultResponse::from_result(
            &node, CLUSTER.deschedule_node(node).await.map(|_| None)))
    }

    Json::from(resp).into_response()
}


