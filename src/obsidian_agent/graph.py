from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from obsidian_agent.config import AppConfig
from obsidian_agent.tools import create_note_tools


class AgentState(TypedDict, total=False):
    question: str
    needs_tool: bool
    search_query: str
    search_results: list[dict[str, str]]
    selected_files: list[str]
    notes_content: list[dict[str, str]]
    iterations: int
    final_answer: str
    messages: list[BaseMessage]
    tool_calls: list[dict[str, Any]]


def create_chat_model(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model.model_name,
        base_url=config.model.base_url,
        api_key=config.model.api_key,
        temperature=config.model.temperature,
    )


def build_graph(config: AppConfig, model: BaseChatModel | None = None):
    note_tools = create_note_tools(
        config.vault_path,
        max_results=config.search.max_results,
    )
    tool_by_name = {item.name: item for item in note_tools}
    llm = model or create_chat_model(config)
    llm_with_tools = llm.bind_tools(note_tools)

    def classify_question(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        iterations = state.get("iterations", 0)
        search_query = state.get("search_query") or question

        if iterations > 0 and not state.get("search_results"):
            search_query = f"{question} Obsidian markdown note"

        needs_tool = _looks_like_vault_question(question)
        return {
            "needs_tool": needs_tool,
            "search_query": search_query,
            "iterations": iterations,
        }

    def search_node(state: AgentState) -> dict[str, Any]:
        messages = [
            SystemMessage(
                content=(
                    "You search a local Obsidian vault. Call search_notes exactly once "
                    "with the best concise keyword query for the user question."
                )
            ),
            HumanMessage(content=state["search_query"]),
        ]
        response = llm_with_tools.invoke(messages)
        tool_calls = list(response.tool_calls)

        if not tool_calls:
            tool_calls = [
                {
                    "name": "search_notes",
                    "args": {"query": state["search_query"]},
                    "id": "fallback-search",
                }
            ]

        search_results: list[dict[str, str]] = []
        tool_messages: list[ToolMessage] = []
        recorded_calls: list[dict[str, Any]] = []

        for call in tool_calls:
            name = call["name"]
            args = call.get("args") or {}
            if name != "search_notes":
                continue
            result = tool_by_name[name].invoke(args)
            search_results = result
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call.get("id", name)))
            recorded_calls.append({"name": name, "args": args, "result_count": len(result)})

        return {
            "search_results": search_results,
            "selected_files": [item["file_path"] for item in search_results],
            "iterations": state.get("iterations", 0) + 1,
            "messages": state.get("messages", []) + [response] + tool_messages,
            "tool_calls": state.get("tool_calls", []) + recorded_calls,
        }

    def read_node(state: AgentState) -> dict[str, Any]:
        notes_content: list[dict[str, str]] = []
        recorded_calls: list[dict[str, Any]] = []

        for file_path in state.get("selected_files", []):
            result = tool_by_name["read_note"].invoke({"file_path": file_path})
            notes_content.append(result)
            recorded_calls.append({"name": "read_note", "args": {"file_path": file_path}})

        return {
            "notes_content": notes_content,
            "tool_calls": state.get("tool_calls", []) + recorded_calls,
        }

    def summarize_node(state: AgentState) -> dict[str, str]:
        notes = state.get("notes_content", [])
        if notes:
            context = "\n\n".join(
                f"## {note['file_path']}\n{note['content']}" for note in notes
            )
            prompt = (
                "请基于以下 Obsidian Markdown 笔记回答用户问题。"
                "输出结构化总结，注明依据来自哪些文件；如果信息不足，请直接说明。\n\n"
                f"用户问题：{state['question']}\n\n笔记内容：\n{context}"
            )
        else:
            prompt = (
                "请直接回答用户问题。若问题要求查询 Obsidian 笔记但没有找到相关内容，"
                "请说明没有在当前 vault 中找到足够信息。\n\n"
                f"用户问题：{state['question']}"
            )

        response = llm.invoke([HumanMessage(content=prompt)])
        return {"final_answer": response.content}

    def route_after_classify(state: AgentState) -> Literal["search_node", "summarize_node"]:
        return "search_node" if state.get("needs_tool") else "summarize_node"

    def route_after_search(state: AgentState) -> Literal["classify_question", "read_node"]:
        if (
            not state.get("search_results")
            and state.get("iterations", 0) < config.search.max_search_iterations
        ):
            return "classify_question"
        return "read_node"

    graph = StateGraph(AgentState)
    graph.add_node("classify_question", classify_question)
    graph.add_node("search_node", search_node)
    graph.add_node("read_node", read_node)
    graph.add_node("summarize_node", summarize_node)
    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges(
        "classify_question",
        route_after_classify,
        {"search_node": "search_node", "summarize_node": "summarize_node"},
    )
    graph.add_conditional_edges(
        "search_node",
        route_after_search,
        {"classify_question": "classify_question", "read_node": "read_node"},
    )
    graph.add_edge("read_node", "summarize_node")
    graph.add_edge("summarize_node", END)
    return graph.compile()


def run_agent(question: str, config: AppConfig, model: BaseChatModel | None = None) -> AgentState:
    graph = build_graph(config, model=model)
    return graph.invoke(
        {
            "question": question,
            "iterations": 0,
            "messages": [],
            "tool_calls": [],
        }
    )


def _looks_like_vault_question(question: str) -> bool:
    lowered = question.casefold()
    indicators = [
        "obsidian",
        "vault",
        "笔记",
        "我的",
        "我在",
        "总结我",
        "markdown",
    ]
    return any(item in lowered for item in indicators)
