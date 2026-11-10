"""A retried subgraph starts over and asks its questions again.

A retry reruns a node from its first line. When the node is a subgraph, the
retry runs the subgraph from its start in the checkpoint namespace the failed
attempt used, so the answer to the new question is the one a later resume reads.
"""

import sys
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from typing_extensions import TypedDict

from langgraph._internal._constants import CONFIG_KEY_STRICT_RESUME
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Durability, RetryPolicy, interrupt

pytestmark = pytest.mark.anyio
NEEDS_CONTEXTVARS = pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="Python 3.11+ is required for async contextvars support",
)

RETRY_ONCE = RetryPolicy(max_attempts=2, initial_interval=0, retry_on=RuntimeError)


class State(TypedDict, total=False):
    amount: int


def _config(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


def _ask_then_check(
    checkpointer: BaseCheckpointSaver,
) -> tuple[CompiledStateGraph, list[int]]:
    """A retried subgraph that asks a question and then calls a flaky service.

    The service fails on its first call only. Returns the graph and the list of
    amounts the service was called with.
    """
    calls: list[int] = []

    def ask(state: State) -> State:
        return {"amount": interrupt("How much?")}

    def check(state: State) -> State:
        calls.append(state["amount"])
        if len(calls) == 1:
            raise RuntimeError("service unavailable")
        return {"amount": state["amount"]}

    attempt = (
        StateGraph(State)
        .add_node("ask", ask)
        .add_node("check", check)
        .add_edge(START, "ask")
        .add_edge("ask", "check")
        .add_edge("check", END)
        .compile()
    )
    graph = (
        StateGraph(State)
        .add_node("attempt", attempt, retry_policy=RETRY_ONCE)
        .add_edge(START, "attempt")
        .add_edge("attempt", END)
        .compile(checkpointer=checkpointer)
    )
    return graph, calls


def _fail_in_second_node(
    checkpointer: BaseCheckpointSaver,
) -> tuple[CompiledStateGraph, list[str]]:
    """A retried subgraph whose second node fails once and needs no input."""
    ran: list[str] = []

    def first(state: State) -> State:
        ran.append("first")
        return {"amount": 1}

    def second(state: State) -> State:
        ran.append("second")
        if ran.count("second") == 1:
            raise RuntimeError("transient failure")
        return {"amount": 2}

    sub = (
        StateGraph(State)
        .add_node("first", first)
        .add_node("second", second)
        .add_edge(START, "first")
        .add_edge("first", "second")
        .add_edge("second", END)
        .compile()
    )
    graph = (
        StateGraph(State)
        .add_node("sub", sub, retry_policy=RETRY_ONCE)
        .add_edge(START, "sub")
        .add_edge("sub", END)
        .compile(checkpointer=checkpointer)
    )
    return graph, ran


def test_a_retried_subgraph_asks_again_and_uses_the_new_answer(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, calls = _ask_then_check(sync_checkpointer)
    config = _config("ask-again")

    result = graph.invoke({}, config, durability=durability)
    assert [i.value for i in result["__interrupt__"]] == ["How much?"]

    # The service fails after 5, so the retry asks the question again.
    result = graph.invoke(Command(resume=5), config, durability=durability)
    assert [i.value for i in result["__interrupt__"]] == ["How much?"]

    # The answer to the new question is the one that counts.
    result = graph.invoke(Command(resume=7), config, durability=durability)

    assert "__interrupt__" not in result
    assert result == {"amount": 7}
    assert calls == [5, 7]


@NEEDS_CONTEXTVARS
async def test_a_retried_subgraph_asks_again_and_uses_the_new_answer_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, calls = _ask_then_check(async_checkpointer)
    config = _config("ask-again-async")

    result = await graph.ainvoke({}, config, durability=durability)
    assert [i.value for i in result["__interrupt__"]] == ["How much?"]

    result = await graph.ainvoke(Command(resume=5), config, durability=durability)
    assert [i.value for i in result["__interrupt__"]] == ["How much?"]

    result = await graph.ainvoke(Command(resume=7), config, durability=durability)

    assert "__interrupt__" not in result
    assert result == {"amount": 7}
    assert calls == [5, 7]


def test_a_retried_subgraph_starts_from_its_first_node(
    sync_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, ran = _fail_in_second_node(sync_checkpointer)

    result = graph.invoke({}, _config("restart"), durability=durability)

    assert result == {"amount": 2}
    assert ran == ["first", "second", "first", "second"]


@NEEDS_CONTEXTVARS
async def test_a_retried_subgraph_starts_from_its_first_node_async(
    async_checkpointer: BaseCheckpointSaver, durability: Durability
) -> None:
    graph, ran = _fail_in_second_node(async_checkpointer)

    result = await graph.ainvoke({}, _config("restart-async"), durability=durability)

    assert result == {"amount": 2}
    assert ran == ["first", "second", "first", "second"]


def test_a_retried_subgraph_asks_a_new_question_in_strict_resume(
    sync_checkpointer: BaseCheckpointSaver,
) -> None:
    """Strict resume answers each question by its own id.

    The retry asks a new question with a new id, and the answer to that one is
    used.
    """
    graph, calls = _ask_then_check(sync_checkpointer)
    config = {"configurable": {"thread_id": "strict", CONFIG_KEY_STRICT_RESUME: True}}
    graph.invoke({}, config)
    (first,) = graph.get_state(config).interrupts

    graph.invoke(Command(resume={first.id: 5}), config)
    (second,) = graph.get_state(config).interrupts
    result = graph.invoke(Command(resume={second.id: 7}), config)

    assert second.id != first.id
    assert "__interrupt__" not in result
    assert result == {"amount": 7}
    assert calls == [5, 7]


@NEEDS_CONTEXTVARS
async def test_a_retried_subgraph_asks_a_new_question_in_strict_resume_async(
    async_checkpointer: BaseCheckpointSaver,
) -> None:
    graph, calls = _ask_then_check(async_checkpointer)
    config = {
        "configurable": {"thread_id": "strict-async", CONFIG_KEY_STRICT_RESUME: True}
    }
    await graph.ainvoke({}, config)
    (first,) = (await graph.aget_state(config)).interrupts

    await graph.ainvoke(Command(resume={first.id: 5}), config)
    (second,) = (await graph.aget_state(config)).interrupts
    result = await graph.ainvoke(Command(resume={second.id: 7}), config)

    assert second.id != first.id
    assert "__interrupt__" not in result
    assert result == {"amount": 7}
    assert calls == [5, 7]
