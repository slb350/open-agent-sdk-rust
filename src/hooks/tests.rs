
    #[tokio::test]
    async fn test_hook_decision_builders() {
        let continue_dec = HookDecision::continue_();
        assert!(continue_dec.continue_execution);
        assert!(continue_dec.reason.is_none());

        let block_dec = HookDecision::block("test");
        assert!(!block_dec.continue_execution);
        assert_eq!(block_dec.reason, Some("test".to_string()));

        let modify_dec = HookDecision::modify_input(json!({"test": 1}), "modified");
        assert!(modify_dec.continue_execution);
        assert!(modify_dec.modified_input.is_some());
    }

    #[tokio::test]
    async fn test_pre_tool_use_hook() {
        let hooks = Hooks::new().add_pre_tool_use(|event| async move {
            if event.tool_name == "dangerous" {
                return Some(HookDecision::block("blocked"));
            }
            None
        });

        let event = PreToolUseEvent::new(
            "dangerous".to_string(),
            json!({}),
            "id1".to_string(),
            vec![],
        );

        let decision = hooks.execute_pre_tool_use(event).await;
        assert!(decision.is_some());
        assert!(!decision.unwrap().continue_execution);
    }

    #[tokio::test]
    async fn test_post_tool_use_hook() {
        let hooks = Hooks::new().add_post_tool_use(|_event| async move { None });

        let event = PostToolUseEvent::new(
            "test".to_string(),
            json!({}),
            "id1".to_string(),
            json!({"result": "ok"}),
            vec![],
        );

        // Should not panic
        hooks.execute_post_tool_use(event).await;
    }

    #[tokio::test]
    async fn test_user_prompt_submit_hook() {
        let hooks = Hooks::new().add_user_prompt_submit(|event| async move {
            if event.prompt.contains("DELETE") {
                return Some(HookDecision::block("dangerous prompt"));
            }
            None
        });

        let event = UserPromptSubmitEvent::new("DELETE all files".to_string(), vec![]);

        let decision = hooks.execute_user_prompt_submit(event).await;
        assert!(decision.is_some());
        assert!(!decision.unwrap().continue_execution);
    }
