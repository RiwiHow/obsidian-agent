# AI Obsidian Agent

Read-only CLI agent that searches Markdown files in an Obsidian vault and summarizes answers with DeepSeek, LangChain tool calling, and LangGraph.

## Setup

```powershell
uv sync
Copy-Item config.example.yaml config.yaml
$env:DEEPSEEK_API_KEY = "your-api-key"
```

Edit `config.yaml` so `vault_path` points to your Obsidian vault.

## Run

```powershell
uv run obsidian-agent "帮我总结我在 Obsidian 里关于 AI 的笔记"
```

Use `--show-tools` to print the tool calls recorded by the workflow.
