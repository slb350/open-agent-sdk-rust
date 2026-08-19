    use futures::StreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::time::{Duration, sleep};

    fn sse_chunk(content: Option<&str>, finish_reason: Option<&str>) -> String {
        serde_json::json!({
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test",
            "choices": [{
                "index": 0,
                "delta": { "content": content },
                "finish_reason": finish_reason,
            }],
        })
        .to_string()
    }

    async fn write_http_chunk(
        socket: &mut tokio::net::TcpStream,
        payload: &[u8],
    ) -> std::io::Result<()> {
        let header = format!("{:X}\r\n", payload.len());
        socket.write_all(header.as_bytes()).await?;
        socket.write_all(payload).await?;
        socket.write_all(b"\r\n").await?;
        socket.flush().await
    }

    #[tokio::test]
    async fn test_parse_sse_stream_buffers_transport_fragments_and_all_events() {
        let first_event = format!("data: {}\n\n", sse_chunk(Some("Héllo"), None));
        let split_at = first_event
            .find('é')
            .expect("test event contains multibyte content")
            + 1;
        let coalesced_events = format!(
            "data: {}\n\ndata: {}\n\ndata: [DONE]\n\n",
            sse_chunk(Some(" world"), None),
            sse_chunk(None, Some("stop")),
        )
        .into_bytes();

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind loopback test server");
        let address = listener.local_addr().expect("read loopback address");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept test client");
            let mut request = [0_u8; 1_024];
            let request_len = socket
                .read(&mut request)
                .await
                .expect("read test request");
            assert!(request_len > 0, "test client sent an HTTP request");
            socket
                .write_all(
                    b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\nconnection: close\r\n\r\n",
                )
                .await
                .expect("write response headers");

            let (first_fragment, second_fragment) = first_event.as_bytes().split_at(split_at);
            write_http_chunk(&mut socket, first_fragment)
                .await
                .expect("write first fragment");
            sleep(Duration::from_millis(10)).await;
            write_http_chunk(&mut socket, second_fragment)
                .await
                .expect("write second fragment");
            sleep(Duration::from_millis(10)).await;
            write_http_chunk(&mut socket, &coalesced_events)
                .await
                .expect("write coalesced events");
            socket
                .write_all(b"0\r\n\r\n")
                .await
                .expect("finish chunked response");
            socket.shutdown().await.expect("close test response");
        });

        let response = reqwest::get(format!("http://{address}"))
            .await
            .expect("request loopback event stream");
        let parsed = parse_sse_stream(response).collect::<Vec<_>>().await;
        server.await.expect("test server task completes");

        let chunks = parsed
            .into_iter()
            .collect::<Result<Vec<_>>>()
            .expect("all fragmented and coalesced events parse");
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].choices[0].delta.content.as_deref(), Some("Héllo"));
        assert_eq!(chunks[1].choices[0].delta.content.as_deref(), Some(" world"));
        assert_eq!(
            chunks[2].choices[0].finish_reason.as_deref(),
            Some("stop")
        );
    }
