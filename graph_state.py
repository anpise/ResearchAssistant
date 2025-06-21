from typing import List, Optional, TypedDict
from typing_extensions import Annotated
from langchain_core.messages import BaseMessage
import operator

class State(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str  # Routing decision
    agent_name: Optional[str]  # Current agent name

class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str  # Instructions for Supervisor agent
    agent_name: Optional[str]  # Current agent name

class DocWritingState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next: str  # Instructions for Supervisor agent
    agent_name: Optional[str]  # Current agent name
    current_files: str  # Currently working files 