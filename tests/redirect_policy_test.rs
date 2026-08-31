mod common;

use common::{header_values, read_request};
use open_agent::{AgentOptions, Client, Error, query};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::task::JoinHandle;
use tokio::time::{Duration, timeout};

const SENTINEL_TOKEN: &str = "tenant-secret-must-stay-at-origin";
const REDIRECT_BODY: &str = "redirects are not model responses";

fn redirect_response(address: std::net::SocketAddr) -> String {
    format!(
        "HTTP/1.1 307 Temporary Redirect\r\nlocation: http://{address}/redirected\r\ncontent-type: text/plain\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{REDIRECT_BODY}",
        REDIRECT_BODY.len()
    )
}

async fn start_redirect_pair() -> (String, JoinHandle<String>, JoinHandle<Option<String>>) {
    let target_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback redirect target");
    let target_address = target_listener
        .local_addr()
        .expect("read redirect target address");
    let target = tokio::spawn(async move {
        let accepted = timeout(Duration::from_secs(1), target_listener.accept()).await;
        let (mut socket, _) = match accepted {
            Err(_) => return None,
            Ok(Ok(connection)) => connection,
            Ok(Err(error)) => panic!("accept redirect target request: {error}"),
        };

        let request = timeout(Duration::from_secs(1), read_request(&mut socket))
            .await
            .expect("redirect target request is complete before timeout");
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
            )
            .await
            .expect("write redirect target response");
        socket.shutdown().await.expect("close redirect target");
        Some(String::from_utf8(request).expect("redirected request is valid UTF-8"))
    });

    let origin_listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback origin");
    let origin_address = origin_listener.local_addr().expect("read origin address");
    let origin = tokio::spawn(async move {
        let (mut socket, _) = origin_listener
            .accept()
            .await
            .expect("accept origin request");
        let request = read_request(&mut socket).await;
        let response = redirect_response(target_address);
        socket
            .write_all(response.as_bytes())
            .await
            .expect("write origin redirect response");
        socket.shutdown().await.expect("close origin response");
        String::from_utf8(request).expect("origin request is valid UTF-8")
    });

    (format!("http://{origin_address}"), origin, target)
}

async fn start_same_origin_redirect() -> (String, JoinHandle<(String, Option<String>)>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback origin");
    let address = listener.local_addr().expect("read origin address");
    let server = tokio::spawn(async move {
        let (mut origin_socket, _) = listener.accept().await.expect("accept origin request");
        let origin_request = read_request(&mut origin_socket).await;
        let response = redirect_response(address);
        origin_socket
            .write_all(response.as_bytes())
            .await
            .expect("write same-origin redirect response");
        origin_socket
            .shutdown()
            .await
            .expect("close same-origin response");

        let redirected_request = match timeout(Duration::from_secs(1), listener.accept()).await {
            Err(_) => None,
            Ok(Ok((mut socket, _))) => {
                let request = timeout(Duration::from_secs(1), read_request(&mut socket))
                    .await
                    .expect("same-origin redirect request is complete before timeout");
                Some(
                    String::from_utf8(request)
                        .expect("same-origin redirected request is valid UTF-8"),
                )
            }
            Ok(Err(error)) => panic!("accept same-origin redirect request: {error}"),
        };

        (
            String::from_utf8(origin_request).expect("origin request is valid UTF-8"),
            redirected_request,
        )
    });

    (format!("http://{address}"), server)
}

fn assert_redirect_not_contacted(target_request: Option<&str>) {
    if let Some(request) = target_request {
        assert_eq!(header_values(request, "X-Tenant-Token"), [SENTINEL_TOKEN]);
        panic!("redirect target received the model request:\n{request}");
    }
}

fn assert_origin_redirect_error(error: Option<Error>) {
    let error = error.expect("origin 307 must surface as an API error");
    assert_eq!(error.status_code(), Some(307));
    assert!(
        error.to_string().contains(REDIRECT_BODY),
        "API error must retain the origin response body: {error}"
    );
}

#[tokio::test]
async fn query_rejects_redirect_without_forwarding_custom_headers() {
    let (base_url, origin, target) = start_redirect_pair().await;
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url(base_url)
        .header("X-Tenant-Token", SENTINEL_TOKEN)
        .build()
        .expect("valid options");

    let error = query("hello", &options).await.err();
    let origin_request = origin.await.expect("origin task completes");
    let target_request = target.await.expect("redirect target task completes");

    assert_eq!(
        header_values(&origin_request, "X-Tenant-Token"),
        [SENTINEL_TOKEN]
    );
    assert_redirect_not_contacted(target_request.as_deref());
    assert_origin_redirect_error(error);
}

#[tokio::test]
async fn client_rejects_redirect_without_forwarding_custom_headers() {
    let (base_url, origin, target) = start_redirect_pair().await;
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url(base_url)
        .header("X-Tenant-Token", SENTINEL_TOKEN)
        .build()
        .expect("valid options");
    let mut client = Client::new(options).expect("construct SDK client");

    let error = client.send("hello").await.err();
    let origin_request = origin.await.expect("origin task completes");
    let target_request = target.await.expect("redirect target task completes");

    assert_eq!(
        header_values(&origin_request, "X-Tenant-Token"),
        [SENTINEL_TOKEN]
    );
    assert_redirect_not_contacted(target_request.as_deref());
    assert_origin_redirect_error(error);
}

#[tokio::test]
async fn same_origin_redirect_is_rejected_without_a_second_request() {
    let (base_url, server) = start_same_origin_redirect().await;
    let options = AgentOptions::builder()
        .model("test-model")
        .base_url(base_url)
        .header("X-Tenant-Token", SENTINEL_TOKEN)
        .build()
        .expect("valid options");

    let error = query("hello", &options).await.err();
    let (origin_request, redirected_request) = server.await.expect("origin task completes");

    assert_eq!(
        header_values(&origin_request, "X-Tenant-Token"),
        [SENTINEL_TOKEN]
    );
    assert_redirect_not_contacted(redirected_request.as_deref());
    assert_origin_redirect_error(error);
}
