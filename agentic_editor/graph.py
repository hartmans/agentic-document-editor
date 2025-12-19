import operator
import typing
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage
import langchain_core.language_models
import dataclasses
import hartmans_ai
from .document import DocumentState

llm: langchain_core.language_models.BaseChatModel = hartmans_ai.local_llamacpp_model.build_langchain_model()

@dataclasses.dataclass
class DocumentUpdate:
    offset: int
    length: int
    new_text: str

@dataclasses.dataclass
class GraphState:

    document: DocumentState|None
    document_update: DocumentUpdate|None
    messages: typing.Annotated[list[BaseMessage], operator.add] = dataclasses.field(default_factory=lambda: [])
