from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from obsidian_agent.config import AppConfig, ModelConfig, SearchConfig
from obsidian_agent.graph import run_agent


def _config(vault: Path, iterations: int = 2) -> AppConfig:
    return AppConfig(
        vault_path=vault,
        model=ModelConfig(),
        search=SearchConfig(max_results=5, max_search_iterations=iterations),
    )


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any):
        return self


def test_obsidian_question_triggers_search_and_read(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "ai.md").write_text("AI note content", encoding="utf-8")
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_notes",
                        "args": {"query": "AI"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(content="总结完成"),
        ]
    )

    result = run_agent("帮我总结我在 Obsidian 里关于 AI 的笔记", _config(vault), model=model)

    assert result["final_answer"] == "总结完成"
    assert [call["name"] for call in result["tool_calls"]] == ["search_notes", "read_note"]
    assert result["notes_content"][0]["file_path"] == "ai.md"


def test_general_question_can_skip_tools(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    model = ToolCallingFakeModel(responses=[AIMessage(content="普通回答")])

    result = run_agent("Python 是什么？", _config(vault), model=model)

    assert result["final_answer"] == "普通回答"
    assert result["tool_calls"] == []


def test_search_failure_retries_without_infinite_loop(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_notes",
                        "args": {"query": "notfound"},
                        "id": "call-1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "search_notes",
                        "args": {"query": "still missing"},
                        "id": "call-2",
                    }
                ],
            ),
            AIMessage(content="没有找到足够信息"),
        ]
    )

    result = run_agent("帮我查我的笔记里不存在的主题", _config(vault, iterations=2), model=model)

    search_calls = [call for call in result["tool_calls"] if call["name"] == "search_notes"]
    assert len(search_calls) == 2
    assert result["final_answer"] == "没有找到足够信息"
