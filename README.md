# AI Obsidian Agent

Read-only CLI agent that retrieves relevant Markdown chunks from an Obsidian vault with local RAG, then summarizes answers with DeepSeek, LangChain tool calling, and LangGraph.

## Setup

```powershell
uv sync
Copy-Item config.example.yaml config.yaml
```

Edit `config.yaml` so `vault_path` points to your Obsidian vault and `model.api_key` contains your DeepSeek API key.

## Run

```powershell
uv run obsidian-agent "帮我总结我在 Obsidian 里关于 AI 的笔记"
```

Use `--show-tools` to print the tool calls recorded by the workflow.
