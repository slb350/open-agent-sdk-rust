use futures::StreamExt;
use open_agent::{AgentOptions, ApiProtocol, Client, Error, query};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

async fn start_request_capture() -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind loopback request capture");
    let address = listener.local_addr().expect("read loopback address");
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.expect("accept SDK request");
        let request = read_request(&mut socket).await;
        socket
            .write_all(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
            )
            .await
            .expect("write empty event stream response");
        socket.shutdown().await.expect("close test response");
        String::from_utf8(request).expect("HTTP request is valid UTF-8")
    });

    (format!("http://{address}"), server)
}

async fn read_request(socket: &mut TcpStream) -> Vec<u8> {
    let mut request = Vec::new();
    let mut buffer = [0_u8; 4_096];

    loop {
        let read = socket.read(&mut buffer).await.expect("read SDK request");
        assert!(read > 0, "SDK closed the request before sending its body");
        request.extend_from_slice(&buffer[..read]);

        let Some(header_end) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") else {
            continue;
        };
        let body_start = header_end + 4;
        let headers = String::from_utf8_lossy(&request[..header_end]);
        let content_length = headers
            .lines()
            .filter_map(|line| line.split_once(':'))
            .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
            .map(|(_, value)| {
                value
                    .trim()
                    .parse::<usize>()
                    .expect("request content-length is numeric")
            })
            .unwrap_or_default();

        if request.len() >= body_start + content_length {
            return request;
        }
    }
}

async fn capture_query_request(configure: impl FnOnce(String) -> AgentOptions) -> String {
    let (base_url, server) = start_request_capture().await;
    let options = configure(base_url);
    let mut stream = query("hello", &options)
        .await
        .expect("send one-shot SDK request");
    while let Some(event) = stream.next().await {
        event.expect("decode empty event stream");
    }
    server.await.expect("request capture task completes")
}

async fn capture_client_request(configure: impl FnOnce(String) -> AgentOptions) -> String {
    let (base_url, server) = start_request_capture().await;
    let options = configure(base_url);
    let mut client = Client::new(options).expect("construct SDK client");
    client.send("hello").await.expect("send client request");
    while client
        .receive()
        .await
        .expect("decode empty event stream")
        .is_some()
    {}
    server.await.expect("request capture task completes")
}

fn header_values(request: &str, expected_name: &str) -> Vec<String> {
    request
        .lines()
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case(expected_name))
        .map(|(_, value)| value.trim().to_string())
        .collect()
}

#[tokio::test]
async fn custom_header_arrives_on_query_request() {
    let request = capture_query_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .header("X-Title", "Request attribution")
            .build()
            .expect("valid options")
    })
    .await;

    assert_eq!(header_values(&request, "X-Title"), ["Request attribution"]);
    assert!(header_values(&request, "User-Agent").is_empty());
}

#[tokio::test]
async fn caller_authorization_replaces_sdk_default_without_a_duplicate() {
    let request = capture_query_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .api_key("SDK credential")
            .header("authorization", "Caller credential")
            .build()
            .expect("valid options")
    })
    .await;

    assert_eq!(
        header_values(&request, "Authorization"),
        ["Caller credential"]
    );
}

#[tokio::test]
async fn empty_api_key_omits_sdk_auth_for_both_protocols() {
    let openai_request = capture_query_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .api_key("")
            .build()
            .expect("valid options")
    })
    .await;
    assert!(header_values(&openai_request, "Authorization").is_empty());

    let anthropic_request = capture_query_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .api_key("")
            .protocol(ApiProtocol::Anthropic)
            .build()
            .expect("valid options")
    })
    .await;
    assert!(header_values(&anthropic_request, "x-api-key").is_empty());
}

#[tokio::test]
async fn unrelated_custom_header_preserves_anthropic_version() {
    let request = capture_query_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .protocol(ApiProtocol::Anthropic)
            .header("X-Title", "Request attribution")
            .build()
            .expect("valid options")
    })
    .await;

    assert_eq!(header_values(&request, "X-Title"), ["Request attribution"]);
    assert_eq!(header_values(&request, "anthropic-version"), ["2023-06-01"]);
}

#[tokio::test]
async fn repeated_header_name_replaces_case_insensitively_on_client_request() {
    let request = capture_client_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .header("x-route", "first")
            .header("X-Route", "second")
            .build()
            .expect("valid options")
    })
    .await;

    assert_eq!(header_values(&request, "X-Route"), ["second"]);
}

#[tokio::test]
async fn caller_can_override_every_anthropic_sdk_header() {
    let request = capture_query_request(|base_url| {
        AgentOptions::builder()
            .model("test-model")
            .base_url(base_url)
            .api_key("SDK credential")
            .protocol(ApiProtocol::Anthropic)
            .header("X-API-Key", "Caller credential")
            .header("Anthropic-Version", "caller-version")
            .header("content-type", "application/custom+json")
            .build()
            .expect("valid options")
    })
    .await;

    assert_eq!(header_values(&request, "x-api-key"), ["Caller credential"]);
    assert_eq!(
        header_values(&request, "anthropic-version"),
        ["caller-version"]
    );
    assert_eq!(
        header_values(&request, "Content-Type"),
        ["application/custom+json"]
    );
}

#[test]
fn invalid_header_name_fails_build_with_the_offending_name() {
    let error = AgentOptions::builder()
        .model("test-model")
        .base_url("http://127.0.0.1:1")
        .header("Bad Header", "value")
        .build()
        .expect_err("invalid header name must fail at build time");

    assert!(matches!(error, Error::Config(_)));
    assert!(error.to_string().contains("Bad Header"));
}

#[test]
fn invalid_header_value_fails_build_with_the_offending_name() {
    let error = AgentOptions::builder()
        .model("test-model")
        .base_url("http://127.0.0.1:1")
        .header("X-Bad-Value", "line one\nline two")
        .build()
        .expect_err("invalid header value must fail at build time");

    assert!(matches!(error, Error::Config(_)));
    assert!(error.to_string().contains("X-Bad-Value"));
}
