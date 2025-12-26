import operator
import os
import typing
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Overwrite
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
import langchain_core.language_models
import dataclasses
import hartmans_ai
from .document import DocumentState, DocumentUpdate, apply_document_update
from . import tools
from .llm_logging import ChatMLJsonlCallbackHandler
from pydantic import BaseModel, Field

def _build_llm() -> langchain_core.language_models.BaseChatModel:
    log_path = os.environ.get("LLM_CHATML_LOG_PATH")
    output_path = Path(log_path) if log_path else Path.cwd() / "logs" / "llm_chatml.jsonl"
    handler = ChatMLJsonlCallbackHandler(output_path)
    return hartmans_ai.local_llamacpp_model.build_langchain_model(
        callbacks=[handler],
    )


llm: langchain_core.language_models.BaseChatModel = _build_llm()

FOCUS_RADIUS = 150
TOOLS = [tools.replace_text, tools.reposition_playback]
SYSTEM_PROMPT = '''
You are a document editor assisting a writer in authoring a document. The document will be read to the author. They can interrupt the reading at any point with instructions for you.
If you are instructed to change the document, use the replace_text tool to edit the document.
If you are instructed to jump to a position in the document or continue playback from a particular position use the reposition_playback tool.
If you are instructed to propose text, propose that text in your response; if it is approved then edit the document.
'''

@dataclasses.dataclass
class GraphState:

    document: typing.Optional[str] = None
    document_state: DocumentState|None = None
    position: int|None = None
    chunk_id: int|None = None
    elapsed_ms:int|None = None
    document_update: DocumentUpdate|None = None
    messages: typing.Annotated[list[BaseMessage], operator.add] = dataclasses.field(default_factory=lambda: [])

class ConversationState(BaseModel):
    '''State of the conversation submitted as a single HumanMessage to the model.'''

    messages: list[BaseMessage]
    position: int = Field(description='Position within the document')
    before_text: str = Field(description="Text before position in the document")
    after_text: str = Field(description='Text after position in the document')
    instructions:str = Field(description="Current instructions from the user")
    document: str = Field(description='Complete document being edited')

async def initial_node(state: GraphState)-> dict:
    assert state.document_state
    return {'document': state.document_state.document}

async def llm_call(state: GraphState) -> dict:
    assert state.document
    tool_llm = llm.bind_tools(TOOLS)
    document = state.document
    position = state.position or 0
    before_text = document[max(position-FOCUS_RADIUS, 0):position]
    after_text = document[position:min(position+FOCUS_RADIUS, len(document)-1)]
    for i, message in reversed(list(enumerate(state.messages))):
        # Skip ToolMessages and AiMessages after the most recent HumanMessage; they are part of tool calls.
        if isinstance(message, HumanMessage):
            break
    else: # No HumanMessage Found
        raise ValueError('No HumanMessage found')
    instructions:str = typing.cast(str,state.messages[i].content)
    extra_messages = state.messages[i+1:]
    conversation_state = ConversationState(
        messages=state.messages[:i],
        instructions=instructions,
        before_text=before_text,
        after_text=after_text,
        document=document,
        position=position
    )
    message = HumanMessage(
        content=f'''
Attached is conversation state in a JSON object. The user is editing the document in the *document* property. They gave you instructions contained in the *instructions* property. Those instructions may refer to a position near where the user was reading; the instructions were given as the user was reading at character *position*. Text immediately before *position* is in *before_text* and text after *position* is in *after_text*. Previous conversational interactions can be found in *messages*.content_blocks=

{conversation_state.json()}
    ''')
    return {
        'messages': [await tool_llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            message] + extra_messages)]
        }

async def final(state:GraphState)-> dict:
    ''' 
    Trim out tool calls and results; apply document updates.
    '''
    def filter_message(m: BaseMessage)->bool:
        match m:
            case ToolMessage():
                return False
            case AIMessage() as a if a.tool_calls:
                return False
            case _:
                return True
    messages = [m for m in state.messages if filter_message(m)]
    results:dict[str, typing.Any] = {'messages': Overwrite(messages)}
    results['document'] = None
    assert state.document_state
    if state.document and (state.document != state.document_state.document):
        results['document_state'] = apply_document_update(state.document, state.document_state)
    return results

def our_tool_node() -> ToolNode:
    return ToolNode(TOOLS)

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node('initial_node', initial_node)
    graph.add_node("llm_call", llm_call)
    graph.add_node("our_tool_node", our_tool_node())
    graph.add_node("final", final)
    graph.set_entry_point("initial_node")
    graph.add_edge('initial_node', 'llm_call')
    graph.add_conditional_edges("llm_call", tools_condition, {
        "tools": "our_tool_node",
        END: "final",
    })
    graph.add_edge("our_tool_node", "llm_call")
    graph.set_finish_point("final")
    return graph.compile()
