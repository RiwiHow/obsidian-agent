from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from obsidian_agent.config import load_config
from obsidian_agent.graph import run_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="obsidian-agent")
    parser.add_argument("question", help="Question to answer from your Obsidian vault.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--show-tools", action="store_true", help="Print recorded tool calls.")
    args = parser.parse_args(argv)

    load_dotenv()
    config = load_config(args.config)
    if not config.model.api_key and not os.getenv("DEEPSEEK_API_KEY"):
        print("model.api_key is not set in config.yaml and DEEPSEEK_API_KEY is not set.", file=sys.stderr)
        return 2

    result = run_agent(args.question, config)
    print(result.get("final_answer", ""))

    if args.show_tools:
        print("\nTool calls:")
        for call in result.get("tool_calls", []):
            print(f"- {call}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
