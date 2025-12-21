import operator
import typing
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Overwrite
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage               
import langchain_core.language_models
import dataclasses
import hartmans_ai
from .document import DocumentState, DocumentUpdate, apply_document_update
from . import tools
from pydantic import BaseModel, Field

llm: langchain_core.language_models.BaseChatModel = hartmans_ai.local_llamacpp_model.build_langchain_model()

FOCUS_RADIUS = 150

@dataclasses.dataclass
class GraphState:

    document: DocumentState|None = None
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
    
async def llm_call(state: GraphState) -> dict:
    assert state.document
    document_state = state.document
    document = document_state.document
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
        'messages': [await llm.ainvoke([message] + extra_messages)]
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
    if state.document_update:
        assert state.document
        results['document'] = apply_document_update(state.document_update, state.document)
        results['position'] = state.document_update.offset
        results['document_update'] = None
    return results

def our_tool_node() -> ToolNode:
    return ToolNode([tools.replace_text])

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("llm_call", llm_call)
    graph.add_node("our_tool_node", our_tool_node())
    graph.add_node("final", final)
    graph.set_entry_point("llm_call")
    graph.add_conditional_edges("llm_call", tools_condition, {
        "tools": "our_tool_node",
        END: "final",
    })
    graph.add_edge("our_tool_node", "llm_call")
    graph.set_finish_point("final")
    return graph.compile()
