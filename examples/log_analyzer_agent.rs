//! Log Analyzer Agent - Intelligent log file analysis with pattern detection
//!
//! This agent analyzes application logs to identify errors, patterns, and anomalies.
//! It provides actionable insights to help debug production issues quickly.
//!
//! Usage:
//!     # Analyze a log file
//!     cargo run --example log_analyzer_agent /path/to/app.log
//!
//! Features:
//! - Automatic error and warning detection
//! - Pattern recognition across log entries
//! - Root cause suggestions via LLM
//! - Prioritized recommendations

use open_agent::{AgentOptions, Client, ContentBlock};
use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

/// Log entry with parsed fields
#[derive(Debug, Clone)]
struct LogEntry {
    raw: String,
    level: String,
    message: String,
    line_number: usize,
}

impl LogEntry {
    fn new(raw: String, level: String, message: String, line_number: usize) -> Self {
        Self {
            raw,
            level: level.to_uppercase(),
            message,
            line_number,
        }
    }
}

/// Parses log files
struct LogParser {
    // Common log pattern: timestamp, level, message
    pattern: Regex,
}

impl LogParser {
    fn new() -> Self {
        // Pattern matches: [2025-01-01T12:00:00] ERROR some message
        // Or: 2025-01-01 12:00:00 [ERROR] message
        // Or: ERROR: message
        let pattern = Regex::new(
            r"(?i)(?:\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\s]*)?\s*(?:\[)?(?P<level>ERROR|FATAL|CRITICAL|WARN|WARNING|INFO|DEBUG)(?:\])?\s*:?\s*(?P<message>.*)"
        ).unwrap();

        Self { pattern }
    }

    fn parse_line(&self, line: &str, line_number: usize) -> Option<LogEntry> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        // Try JSON format
        if line.starts_with('{') && line.ends_with('}') {
            if let Ok(json_value) = serde_json::from_str::<Value>(line) {
                let level = json_value
                    .get("level")
                    .or_else(|| json_value.get("severity"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("INFO");

                let message = json_value
                    .get("message")
                    .or_else(|| json_value.get("msg"))
                    .and_then(|v| v.as_str())
                    .unwrap_or(line);

                return Some(LogEntry::new(
                    line.to_string(),
                    level.to_string(),
                    message.to_string(),
                    line_number,
                ));
            }
        }

        // Try regex pattern
        if let Some(caps) = self.pattern.captures(line) {
            let level = caps.name("level").map(|m| m.as_str()).unwrap_or("INFO");
            let message = caps.name("message").map(|m| m.as_str()).unwrap_or(line);

            return Some(LogEntry::new(
                line.to_string(),
                level.to_string(),
                message.to_string(),
                line_number,
            ));
        }

        // Fallback: check for common error keywords
        let lower = line.to_lowercase();
        if lower.contains("error") || lower.contains("exception") || lower.contains("fatal") {
            return Some(LogEntry::new(
                line.to_string(),
                "ERROR".to_string(),
                line.to_string(),
                line_number,
            ));
        } else if lower.contains("warn") {
            return Some(LogEntry::new(
                line.to_string(),
                "WARNING".to_string(),
                line.to_string(),
                line_number,
            ));
        }

        // Default to INFO
        Some(LogEntry::new(
            line.to_string(),
            "INFO".to_string(),
            line.to_string(),
            line_number,
        ))
    }
}

/// Analyzes parsed logs
struct LogAnalyzer {
    entries: Vec<LogEntry>,
    errors: Vec<LogEntry>,
    warnings: Vec<LogEntry>,
}

impl LogAnalyzer {
    fn new(entries: Vec<LogEntry>) -> Self {
        let errors: Vec<LogEntry> = entries
            .iter()
            .filter(|e| matches!(e.level.as_str(), "ERROR" | "FATAL" | "CRITICAL"))
            .cloned()
            .collect();

        let warnings: Vec<LogEntry> = entries
            .iter()
            .filter(|e| matches!(e.level.as_str(), "WARN" | "WARNING"))
            .cloned()
            .collect();

        Self {
            entries,
            errors,
            warnings,
        }
    }

    fn get_summary(&self) -> Value {
        let total = self.entries.len();
        let error_rate = if total > 0 {
            (self.errors.len() as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        // Find error patterns
        let error_patterns = self.find_error_patterns();

        json!({
            "total_entries": total,
            "errors": self.errors.len(),
            "warnings": self.warnings.len(),
            "error_rate": format!("{:.1}%", error_rate),
            "top_error_patterns": error_patterns,
        })
    }

    fn find_error_patterns(&self) -> Vec<(String, usize)> {
        if self.errors.is_empty() {
            return Vec::new();
        }

        let mut patterns: HashMap<String, usize> = HashMap::new();

        let exception_re = Regex::new(r"(\w+(?:Exception|Error))").unwrap();
        let file_re = Regex::new(r"(?:file|at)\s+([^\s:]+\.\w+)").unwrap();

        for error in &self.errors {
            let msg = &error.message;

            // Look for exception names
            if let Some(caps) = exception_re.captures(msg) {
                let exception = caps.get(1).unwrap().as_str().to_string();
                *patterns.entry(exception).or_insert(0) += 1;
                continue;
            }

            // Look for file paths
            if let Some(caps) = file_re.captures(msg) {
                let file = format!("File: {}", caps.get(1).unwrap().as_str());
                *patterns.entry(file).or_insert(0) += 1;
                continue;
            }

            // Generic classification
            let lower = msg.to_lowercase();
            if lower.contains("connection") {
                *patterns.entry("Connection Error".to_string()).or_insert(0) += 1;
            } else if lower.contains("timeout") {
                *patterns.entry("Timeout Error".to_string()).or_insert(0) += 1;
            } else if lower.contains("permission") || lower.contains("denied") {
                *patterns.entry("Permission Error".to_string()).or_insert(0) += 1;
            } else if lower.contains("not found") || lower.contains("404") {
                *patterns.entry("Not Found Error".to_string()).or_insert(0) += 1;
            } else {
                // Use first few words as pattern
                let words: Vec<&str> = msg.split_whitespace().take(5).collect();
                let pattern = words.join(" ");
                if !pattern.is_empty() {
                    *patterns.entry(pattern).or_insert(0) += 1;
                }
            }
        }

        // Convert to sorted vector
        let mut sorted_patterns: Vec<(String, usize)> = patterns.into_iter().collect();
        sorted_patterns.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count, descending
        sorted_patterns.into_iter().take(5).collect()
    }
}

/// Log Analyzer Agent
struct LogAnalyzerAgent {
    options: AgentOptions,
}

impl LogAnalyzerAgent {
    fn new(options: AgentOptions) -> Self {
        Self { options }
    }

    async fn analyze_logs(
        &self,
        log_file: &PathBuf,
        analyzer: &LogAnalyzer,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let summary = analyzer.get_summary();

        // Prepare context for LLM
        let mut context = format!(
            "Analyze these application logs from {} and provide actionable insights.\n\n",
            log_file.display()
        );
        context.push_str(&format!(
            "Log Summary:\n{}\n\n",
            serde_json::to_string_pretty(&summary)?
        ));

        context.push_str("Sample Errors (first 10):\n");
        for (i, error) in analyzer.errors.iter().take(10).enumerate() {
            let display = if error.raw.len() > 150 {
                format!("{}...", &error.raw[..150])
            } else {
                error.raw.clone()
            };
            context.push_str(&format!(
                "{}. Line {}: {}\n",
                i + 1,
                error.line_number,
                display
            ));
        }

        context.push_str(
            r#"

Based on this analysis, provide:
1. Main issues identified (prioritized by severity)
2. Root cause analysis for the top errors
3. Specific recommendations to fix each issue
4. Monitoring suggestions to prevent future issues

Be specific and actionable. If you see patterns like connection errors,
timeout issues, or permission problems, provide concrete steps to diagnose
and resolve them."#,
        );

        let mut client = Client::new(self.options.clone());
        client.send(&context).await?;

        let mut response = String::new();
        while let Some(block) = client.receive().await {
            match block? {
                ContentBlock::Text(text) => {
                    response.push_str(&text.text);
                }
                _ => {}
            }
        }

        Ok(response)
    }

    async fn run(&self, log_file: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
        println!("📁 Analyzing log file: {}", log_file.display());
        println!("{}", "=".repeat(50));

        // Read log file
        let file = File::open(&log_file)?;
        let reader = BufReader::new(file);
        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

        println!("📊 Parsing {} log lines...", lines.len());

        // Parse entries
        let parser = LogParser::new();
        let mut entries = Vec::new();

        for (line_number, line) in lines.iter().enumerate() {
            if let Some(entry) = parser.parse_line(line, line_number + 1) {
                entries.push(entry);
            }
        }

        if entries.is_empty() {
            println!("❌ No valid log entries found");
            return Ok(());
        }

        println!("✓ Parsed {} log entries", entries.len());

        // Analyze logs
        let analyzer = LogAnalyzer::new(entries);
        let summary = analyzer.get_summary();

        // Print summary
        println!("\n📈 Log Summary:");
        println!("  Total entries: {}", summary["total_entries"]);
        println!(
            "  Errors: {} ({})",
            summary["errors"], summary["error_rate"]
        );
        println!("  Warnings: {}", summary["warnings"]);

        if let Some(patterns) = summary["top_error_patterns"].as_array() {
            if !patterns.is_empty() {
                println!("\n🔴 Top Error Patterns:");
                for pattern in patterns {
                    if let Some(arr) = pattern.as_array() {
                        if let (Some(name), Some(count)) = (arr.get(0), arr.get(1)) {
                            println!("  - {}: {} occurrences", name, count);
                        }
                    }
                }
            }
        }

        // Generate analysis
        println!("\n🤖 Generating intelligent analysis...");
        let analysis = self.analyze_logs(&log_file, &analyzer).await?;

        println!("\n{}", "=".repeat(50));
        println!("📋 ANALYSIS REPORT");
        println!("{}", "=".repeat(50));
        println!("{}", analysis);
        println!("{}", "=".repeat(50));

        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <log_file>", args[0]);
        eprintln!("\nExample:");
        eprintln!("  cargo run --example log_analyzer_agent /var/log/app.log");
        std::process::exit(1);
    }

    let log_file = PathBuf::from(&args[1]);

    if !log_file.exists() {
        eprintln!("❌ Log file not found: {}", log_file.display());
        std::process::exit(1);
    }

    // Configuration
    let options = AgentOptions::builder()
        .system_prompt(
            "You are an expert log analyzer and site reliability engineer. \
             You excel at finding patterns in logs, identifying root causes of errors, \
             and providing actionable recommendations. You understand various log formats \
             and can correlate events to find issues. Always be specific and practical \
             in your recommendations.",
        )
        .model("qwen3:8b")
        .base_url("http://localhost:11434/v1")
        .temperature(0.3) // Lower for consistent analysis
        .max_tokens(2000)
        .build()?;

    let agent = LogAnalyzerAgent::new(options);
    agent.run(log_file).await?;

    Ok(())
}
