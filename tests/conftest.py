import asyncio

import pytest
from langchain_core.messages import AIMessage

from agentic_editor.graph import build_graph, build_graph_state


@pytest.fixture
def tool_graph():
    """Run the real editing graph with explicitly scripted LLM responses."""
    def run(document_state, responses: list[AIMessage], *, position=0):
        class ScriptedLLM:
            def __init__(self):
                self.responses = iter(responses)

            def bind_tools(self, tools):
                return self

            async def ainvoke(self, messages):
                try:
                    return next(self.responses)
                except StopIteration:
                    raise AssertionError("Graph requested an unscripted LLM response") from None

        model = ScriptedLLM()
        graph = build_graph(llm=model)
        draft = build_graph_state(document_state, position, "Run the scripted tool calls.")
        result = asyncio.run(graph.ainvoke(draft))
        assert next(model.responses, None) is None, "Graph left scripted responses unused"
        return result

    return run
