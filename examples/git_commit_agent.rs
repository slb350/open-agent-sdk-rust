//! Git Commit Agent - Analyzes staged changes and writes professional commit messages
//!
//! This agent examines your staged git changes and generates well-structured commit
//! messages following conventional commit format (feat/fix/docs/chore/etc).
//!
//! Usage:
//!     # Stage your changes first
//!     git add .
//!
//!     # Run the agent
//!     cargo run --example git_commit_agent
//!
//!     # Agent will analyze changes and suggest a commit message
//!     # You can accept, edit, or regenerate
//!
//! Features:
//! - Analyzes file changes to determine commit type
//! - Writes clear, descriptive commit messages
//! - Follows conventional commit format
//! - Includes breaking change detection
//! - Lists affected files in the body

use open_agent::{AgentOptions, Client, ContentBlock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, Write};
use std::process::Command;

/// Commit message structure
#[derive(Debug, Serialize, Deserialize)]
struct CommitData {
    #[serde(rename = "type")]
    commit_type: String,
    #[serde(default)]
    scope: String,
    subject: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    breaking: String,
}

/// Git Commit Agent
struct GitCommitAgent {
    options: AgentOptions,
}

impl GitCommitAgent {
    fn new(options: AgentOptions) -> Self {
        Self { options }
    }

    /// Execute a git command and return output
    fn run_git_command(&self, args: &[&str]) -> Result<String, Box<dyn std::error::Error>> {
        let mut cmd = Command::new("git");
        cmd.args(args);

        let output = cmd.output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Git command failed: {}", stderr);
            return Ok(String::new());
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Get staged changes with file paths and diff content
    fn get_staged_changes(&self) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
        // Get list of staged files
        let staged_files = self.run_git_command(&["diff", "--cached", "--name-only"])?;

        if staged_files.is_empty() {
            return Ok(HashMap::new());
        }

        let mut changes = HashMap::new();
        for file in staged_files.lines() {
            if !file.is_empty() {
                // Get the diff for this specific file
                let diff = self.run_git_command(&["diff", "--cached", file])?;
                changes.insert(file.to_string(), diff);
            }
        }

        Ok(changes)
    }

    /// Get a summary of staged changes
    fn get_diff_summary(&self) -> Result<String, Box<dyn std::error::Error>> {
        self.run_git_command(&["diff", "--cached", "--stat"])
    }

    /// Create a structured analysis of the changes for the LLM
    fn analyze_changes(
        &self,
        changes: &HashMap<String, String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        if changes.is_empty() {
            return Ok("No staged changes found.".to_string());
        }

        let mut analysis = format!("Staged changes in {} file(s):\n\n", changes.len());
        analysis.push_str(&format!("Summary:\n{}\n\n", self.get_diff_summary()?));

        // Include sample diffs (limit size for context)
        analysis.push_str("Detailed changes:\n");
        let mut total_chars = 0;
        let max_chars = 3000; // Limit context size

        for (file, diff) in changes.iter() {
            if total_chars > max_chars {
                analysis.push_str(&format!(
                    "\n... and {} more files",
                    changes.len() - analysis.matches("---").count()
                ));
                break;
            }

            // Truncate large diffs
            let diff_display = if diff.len() > 500 {
                format!("{}\n... (truncated)", &diff[..500])
            } else {
                diff.clone()
            };

            analysis.push_str(&format!("\n--- {} ---\n{}\n", file, diff_display));
            total_chars += diff_display.len();
        }

        Ok(analysis)
    }

    /// Use LLM to generate a commit message based on changes
    async fn generate_commit_message(
        &self,
        changes: &HashMap<String, String>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let analysis = self.analyze_changes(changes)?;

        if analysis.contains("No staged changes") {
            return Ok(String::new());
        }

        let prompt = format!(
            r#"Analyze these git changes and write a professional commit message.

{}

Based on these changes, write a commit message following these rules:

1. Use conventional commit format: type(scope): description
2. Types to choose from: feat, fix, docs, style, refactor, perf, test, chore, ci, build
3. Keep the first line under 72 characters
4. Add a blank line after the first line
5. Include a bullet list of key changes in the body
6. If there are breaking changes, add "BREAKING CHANGE:" section

Format your response as JSON with these fields:
{{
  "type": "feat|fix|docs|etc",
  "scope": "optional scope like 'auth' or 'api'",
  "subject": "imperative mood description without type prefix",
  "body": "detailed bullet points of changes",
  "breaking": "any breaking changes or empty string"
}}

Focus on WHAT changed and WHY, not just restating the diff."#,
            analysis
        );

        let mut client = Client::new(self.options.clone());
        client.send(&prompt).await?;

        let mut response = String::new();
        while let Some(block) = client.receive().await {
            if let ContentBlock::Text(text) = block? {
                response.push_str(&text.text);
            }
        }

        // Try to parse JSON response
        let mut cleaned = response.trim().to_string();

        // Remove markdown code fences if present
        if cleaned.starts_with("```") {
            cleaned = cleaned.trim_matches('`').to_string();
            if cleaned.starts_with("json") {
                cleaned = cleaned[4..].to_string();
            }
        }

        // Find JSON object boundaries
        if let Some(start) = cleaned.find('{') {
            if let Some(end) = cleaned.rfind('}') {
                if end > start {
                    let json_str = &cleaned[start..=end];
                    match serde_json::from_str::<CommitData>(json_str) {
                        Ok(commit_data) => return Ok(self.format_commit_message(&commit_data)),
                        Err(_) => {
                            // Fallback: return raw response if not valid JSON
                        }
                    }
                }
            }
        }

        // Fallback: return raw response
        Ok(response)
    }

    /// Format the commit message from structured data
    fn format_commit_message(&self, data: &CommitData) -> String {
        let scope_part = if !data.scope.is_empty() {
            format!("({})", data.scope)
        } else {
            String::new()
        };

        let mut message = format!("{}{}: {}", data.commit_type, scope_part, data.subject);

        if !data.body.is_empty() {
            // Ensure body is formatted as bullet points
            let body = data.body.trim();
            let formatted_body = if !body.starts_with('-') {
                // Convert to bullet points if not already
                body.lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| format!("- {}", line.trim()))
                    .collect::<Vec<_>>()
                    .join("\n")
            } else {
                body.to_string()
            };
            message.push_str(&format!("\n\n{}", formatted_body));
        }

        if !data.breaking.is_empty() {
            message.push_str(&format!("\n\nBREAKING CHANGE: {}", data.breaking));
        }

        message
    }

    /// Main agent flow
    async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🔍 Git Commit Agent");
        println!("{}", "=".repeat(50));

        // Check for staged changes
        let changes = self.get_staged_changes()?;

        if changes.is_empty() {
            println!("❌ No staged changes found!");
            println!("\nPlease stage your changes first:");
            println!("  git add <files>");
            println!("  git add .");
            return Ok(());
        }

        println!("✓ Found staged changes in {} file(s)", changes.len());
        println!("\n📊 Change summary:");
        println!("{}", self.get_diff_summary()?);

        println!("\n🤖 Analyzing changes and generating commit message...");

        let mut commit_message = self.generate_commit_message(&changes).await?;

        if commit_message.is_empty() {
            println!("❌ Failed to generate commit message");
            return Ok(());
        }

        // Interactive loop
        loop {
            println!("\n📝 Suggested commit message:");
            println!("{}", "-".repeat(50));
            println!("{}", commit_message);
            println!("{}", "-".repeat(50));

            println!("\nOptions:");
            println!("  [a] Accept and commit");
            println!("  [e] Edit message");
            println!("  [r] Regenerate");
            println!("  [c] Cancel");

            print!("\nYour choice: ");
            io::stdout().flush()?;

            let mut choice = String::new();
            io::stdin().read_line(&mut choice)?;
            let choice = choice.trim().to_lowercase();

            match choice.as_str() {
                "a" => {
                    // Commit with the message
                    let output = Command::new("git")
                        .args(["commit", "-m", &commit_message])
                        .output()?;

                    if output.status.success() {
                        println!("✅ Successfully committed!");
                        println!("{}", String::from_utf8_lossy(&output.stdout));
                    } else {
                        eprintln!(
                            "❌ Commit failed: {}",
                            String::from_utf8_lossy(&output.stderr)
                        );
                    }
                    break;
                }
                "e" => {
                    println!("\nEnter your edited message (end with a line containing only '.'):");
                    let mut lines = Vec::new();
                    loop {
                        let mut line = String::new();
                        io::stdin().read_line(&mut line)?;
                        if line.trim() == "." {
                            break;
                        }
                        lines.push(line.trim_end().to_string());
                    }

                    let edited_message = lines.join("\n");
                    if !edited_message.trim().is_empty() {
                        let output = Command::new("git")
                            .args(["commit", "-m", &edited_message])
                            .output()?;

                        if output.status.success() {
                            println!("✅ Successfully committed with edited message!");
                        } else {
                            eprintln!(
                                "❌ Commit failed: {}",
                                String::from_utf8_lossy(&output.stderr)
                            );
                        }
                    }
                    break;
                }
                "r" => {
                    println!("\n🤖 Regenerating commit message...");
                    commit_message = self.generate_commit_message(&changes).await?;
                }
                "c" => {
                    println!("❌ Cancelled");
                    break;
                }
                _ => {
                    println!("Invalid choice. Please select a, e, r, or c.");
                }
            }
        }

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration - uses defaults for Ollama
    let options = AgentOptions::builder()
        .system_prompt(
            "You are a git commit message expert. You write clear, \
             professional commit messages that follow conventional commit standards. \
             You understand code changes and can identify the type and scope of changes. \
             Always be concise but descriptive.",
        )
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.3) // Lower temperature for consistent formatting
        .max_tokens(500)
        .build()?;

    let agent = GitCommitAgent::new(options);
    agent.run().await?;

    Ok(())
}
