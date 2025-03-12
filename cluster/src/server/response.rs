use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;
use crate::cluster::Error;

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

#[derive(Debug, Serialize)]
pub struct DefaultResponse<'a> {
    node: &'a str,
    success: bool,
    error: Option<ErrResponse<'a>>,
    message: Option<&'a str>,
}

impl<'a> DefaultResponse<'a> {
    pub fn err(node: &'a str, e: ErrResponse<'a>) -> Self {
        DefaultResponse { node, success: false, error: Some(e), message: None }
    }

    pub fn ok(node: &'a str, message: Option<&'a str>) -> Self {
        DefaultResponse { node, success: true, error: None, message }
    }

    pub fn from_result(node: &'a str, result: Result<Option<&'a str>, Error>) -> Self {
        match result {
            Ok(msg) => DefaultResponse::ok(node, msg),
            Err(e) => DefaultResponse::err(node, e.into()),
        }
    }
}

impl IntoResponse for DefaultResponse<'_> {
    fn into_response(self) -> Response {
        Json::from(self).into_response()
    }
}