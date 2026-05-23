"""Tests for Pydantic schema contracts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import (
    Artifact, DecisionOutput, Goal, MemoryItem, Observation, ToolCall,
)


def test_memory_item_serialization():
    item = MemoryItem(
        id="test1", kind="fact", keywords=["hello"],
        descriptor="test fact", value={"key": "val"},
        source="test", run_id="r1",
    )
    data = item.model_dump(mode="json")
    restored = MemoryItem.model_validate(data)
    assert restored.id == "test1"
    assert restored.kind == "fact"


def test_observation_all_done():
    obs = Observation(goals=[
        Goal(id="g1", text="task 1", done=True),
        Goal(id="g2", text="task 2", done=True),
    ])
    assert obs.all_done is True


def test_observation_not_all_done():
    obs = Observation(goals=[
        Goal(id="g1", text="task 1", done=True),
        Goal(id="g2", text="task 2", done=False),
    ])
    assert obs.all_done is False
    assert obs.next_unfinished().id == "g2"


def test_decision_output_is_answer():
    out = DecisionOutput(answer="hello world")
    assert out.is_answer is True
    assert out.is_error is False


def test_decision_output_is_error():
    out = DecisionOutput(is_error=True)
    assert out.is_answer is False
    assert out.is_error is True


def test_decision_output_answer_with_error_flag():
    out = DecisionOutput(answer="[gateway error]", is_error=True)
    assert out.is_answer is False


def test_decision_output_tool_call():
    out = DecisionOutput(tool_call=ToolCall(name="web_search", arguments={"query": "test"}))
    assert out.is_answer is False
    assert out.tool_call.name == "web_search"


def test_artifact_model():
    art = Artifact(id="art:abc123", content_type="text/plain", size_bytes=1024,
                   source="tool:fetch_url", descriptor="test artifact")
    assert art.id.startswith("art:")
