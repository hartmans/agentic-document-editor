import json
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)


def _json_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=True)


def _tool_call_to_chatml(tool_call: Any) -> Dict[str, Any]:
    if isinstance(tool_call, dict):
        name = tool_call.get("name")
        args = tool_call.get("args")
        tool_call_id = tool_call.get("id")
        if "function" in tool_call:
            return tool_call
        return {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": _json_content(args) if args is not None else "",
            },
        }
    name = getattr(tool_call, "name", None)
    args = getattr(tool_call, "args", None)
    tool_call_id = getattr(tool_call, "id", None)
    return {
        "id": tool_call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": _json_content(args) if args is not None else "",
        },
    }


def message_to_chatml(message: BaseMessage) -> Dict[str, Any]:
    match message:
        case SystemMessage():
            return {"role": "system", "content": _json_content(message.content)}
        case HumanMessage():
            return {"role": "user", "content": _json_content(message.content)}
        case ToolMessage():
            payload: Dict[str, Any] = {
                "role": "tool",
                "content": _json_content(message.content),
            }
            if message.name:
                payload["name"] = message.name
            if message.tool_call_id:
                payload["tool_call_id"] = message.tool_call_id
            return payload
        case AIMessage():
            payload = {"role": "assistant", "content": _json_content(message.content)}
            tool_calls = message.tool_calls or []
            if tool_calls:
                payload["tool_calls"] = [_tool_call_to_chatml(tc) for tc in tool_calls]
            return payload
        case _:
            return {"role": "assistant", "content": _json_content(message.content)}


def messages_to_chatml(messages: Iterable[BaseMessage]) -> List[Dict[str, Any]]:
    return [message_to_chatml(message) for message in messages]


class ChatMLJsonlCallbackHandler(BaseCallbackHandler):
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self._pending_inputs: Dict[str, List[List[Dict[str, Any]]]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs: Any) -> Any:
        run_id = str(kwargs.get("run_id", ""))
        if not run_id:
            return None
        inputs = [messages_to_chatml(batch) for batch in messages]
        with self._lock:
            self._pending_inputs[run_id] = inputs
        return None

    def on_llm_end(self, response: Any, **kwargs: Any) -> Any:
        run_id = str(kwargs.get("run_id", ""))
        outputs: List[Dict[str, Any]] = []
        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                if message is not None:
                    outputs.append(message_to_chatml(message))
                else:
                    outputs.append({"role": "assistant", "content": _json_content(generation.text)})
        record = {
            "run_id": run_id,
            "inputs": self._pending_inputs.pop(run_id, []),
            "outputs": outputs,
        }
        with self._lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return None

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> Any:
        run_id = str(kwargs.get("run_id", ""))
        if not run_id:
            return None
        record = {
            "run_id": run_id,
            "inputs": self._pending_inputs.pop(run_id, []),
            "error": str(error),
        }
        with self._lock:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            with self.output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        return None
