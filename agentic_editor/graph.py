import operator
import os
import typing
from pathlib import Path
from langgraph.graph import StateGraph, END
from langgraph.runtime import Runtime
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Overwrite, Command
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
import langchain_core.language_models
import dataclasses
from .document import DocumentState, DocumentUpdate, apply_document_update
from . import tools
from .llm_logging import ChatMLJsonlCallbackHandler
from pydantic import BaseModel, Field

_DEFAULT_LLM_CONFIG = {"model": "ollama:gemma4:31b",
                       'reasoning': False}


def _parse_model_ref(model_ref: str) -> tuple[str, str | None]:
    provider, sep, model_name = model_ref.partition(":")
    if sep and provider:
        return model_name, provider
    return model_ref, None


def _build_llm(
    config: dict[str, typing.Any] | None = None,
) -> langchain_core.language_models.BaseChatModel:
    config = config or {}
    llm_config = config.get("llm") or dict(_DEFAULT_LLM_CONFIG)
    if not isinstance(llm_config, dict):
        raise ValueError("llm config must be a mapping if provided")

    model_name, model_provider = _parse_model_ref(str(llm_config.get("model") or _DEFAULT_LLM_CONFIG["model"]))
    log_path = os.environ.get("LLM_CHATML_LOG_PATH")
    output_path = Path(log_path) if log_path else Path.cwd() / "logs" / "llm_chatml.jsonl"
    handler = ChatMLJsonlCallbackHandler(output_path)
    init_kwargs: dict[str, typing.Any] = {
        "model": model_name,
        "callbacks": [handler],
    }
    if model_provider:
        init_kwargs["model_provider"] = model_provider
    for key, value in llm_config.items():
        if key != "model":
            init_kwargs[key] = value
    return init_chat_model(**init_kwargs)

FOCUS_RADIUS = 150
TOOLS = [tools.replace_text, tools.reposition_playback]
SYSTEM_PROMPT = '''
You are a document editor assisting a writer in authoring a document. The document will be read to the author. They can interrupt the reading at any point with instructions for you. Because the instructions are going through automatic speech recognition, you need to correct capitalization and to account for possessives vs plurals and other areas where speech recognition may make errors.
If you are instructed to change the document, use the replace_text tool to edit the document.
If you are instructed to jump to a position in the document or continue playback from a particular position use the reposition_playback tool.
If you are instructed to propose text, propose that text in your response; if it is approved then edit the document.
If you are instructed to mark text for later editing insert the string [edit] in front of the text.
Once you receive the instructions, determine whether you are asked to change the document. If so, you must use the replace_text tool. If nott, you must not use the replace_text tool.
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

async def _llm_call_with_model(
    llm: langchain_core.language_models.BaseChatModel,
    state: GraphState,
) -> dict:
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

async def our_tool_node(state:GraphState, runtime:Runtime) -> Command | typing.Sequence[Command]:
    node =  ToolNode(TOOLS)
    default_destination = 'llm_call'
    result = await node.ainvoke(state)
    match result:
        case Command(goto=dest) | [Command(goto=dest)] if dest :
            return result # already includes a destination
        case Command(update=update) | [Command(update=update)] if update:
            return Command(goto=default_destination, update=update)
        case Command():
            raise RuntimeError('A command we do not know how to handle; for example Send API')
        case _:
            return Command(goto=default_destination, update=result)

def build_graph(
    config: dict[str, typing.Any] | None = None,
    *,
    llm: langchain_core.language_models.BaseChatModel | None = None,
):
    graph_llm = llm or _build_llm(config)

    async def llm_call(state: GraphState) -> dict:
        return await _llm_call_with_model(graph_llm, state)

    graph = StateGraph(GraphState)
    graph.add_node('initial_node', initial_node)
    graph.add_node("llm_call", llm_call)
    graph.add_node("our_tool_node", our_tool_node)
    graph.add_node("final", final)
    graph.set_entry_point("initial_node")
    graph.add_edge('initial_node', 'llm_call')
    graph.add_conditional_edges("llm_call", tools_condition, {
        "tools": "our_tool_node",
        END: "final",
    })
    graph.set_finish_point("final")
    return graph.compile()
