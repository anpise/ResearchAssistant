import streamlit as st
import re
from typing import List, Dict, Optional, TypedDict, Callable, Any, Literal, Union
from pathlib import Path
from typing_extensions import Annotated
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langgraph.graph import Graph, StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent, Tool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
import operator
from langgraph.checkpoint.memory import MemorySaver
import os
from dotenv import load_dotenv
from langchain.callbacks.base import BaseCallbackHandler
import datetime
import json
import requests
import tiktoken
import uuid

from agents import AgentFactory, create_team_supervisor
from prompts import (
    RESEARCH_SUPERVISOR_PROMPT,
    RESEARCHER_PROMPT,
    EDITOR_PROMPT,
    DOC_WRITER_PROMPT,
)
from config import llm, MODEL_NAME
from tools import tavily_tool, scrape_webpages, create_outline, read_document, write_document, edit_document
from graph_state import State, ResearchState, DocWritingState
from streamlit_callbacks import StreamlitCallbackHandler

# Set page config
st.set_page_config(
    page_title="Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal dark theme CSS
st.markdown("""
<style>
    body, .main {
        background-color: #181818 !important;
        color: #f1f1f1 !important;
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3 {
        color: #00E5FF !important;
    }
    .stButton>button {
        background-color: #222 !important;
        color: #fff !important;
        border-radius: 6px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00E5FF !important;
        color: #181818 !important;
    }
    .stTextArea>div>div>textarea {
        background-color: #222 !important;
        color: #fff !important;
        border-radius: 6px;
        border: 1px solid #333 !important;
    }
    .stTextArea>div>div>textarea:focus {
        border-color: #00E5FF !important;
    }
    .stSlider>div>div>div>div {
        background: #00E5FF !important;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables
load_dotenv()

# Create temporary directory
WORKING_DIRECTORY = Path("./tmp")
WORKING_DIRECTORY.mkdir(exist_ok=True)

# Initialize tools
tavily_tool = TavilySearchResults(k=6)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    def num_tokens_from_string(string: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def clean_text(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        text = soup.get_text(separator=" ").strip()
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        return text

    results = []
    skipped_urls = []
    token_limit = 100000  # Maximum tokens per page

    for url in urls:
        try:
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36",
                },
                timeout=10
            )
            response.raise_for_status()
            
            # Clean the text first
            cleaned_text = clean_text(response.text)
            
            # Check token count
            token_count = num_tokens_from_string(cleaned_text)
            
            if token_count > token_limit:
                skipped_urls.append(f"{url} (tokens: {token_count})")
                continue
                
            results.append(
                f'<Document name="{BeautifulSoup(response.text, "html.parser").title.string if BeautifulSoup(response.text, "html.parser").title else url.strip()}">\n{cleaned_text}\n</Document>'
            )
        except Exception as e:
            skipped_urls.append(f"{url} (error: {str(e)})")
            continue

    # Prepare the final output
    output = "\n\n".join(results)
    if skipped_urls:
        output += "\n\nSkipped URLs (too large or error):\n" + "\n".join(skipped_urls)
    
    return output

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
    session_folder: str = None
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    folder = session_folder if session_folder else "research_sessions/final"
    os.makedirs(folder, exist_ok=True)
    
    with open(os.path.join(folder, file_name), "w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to read the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
    session_folder: str = None
) -> str:
    """Read the specified document."""
    folder = session_folder if session_folder else "research_sessions/final"
    with open(os.path.join(folder, file_name), "r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
    session_folder: str = None
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    folder = session_folder if session_folder else "research_sessions/final"
    os.makedirs(folder, exist_ok=True)
    
    with open(os.path.join(folder, file_name), "w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "File path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
    session_folder: str = None
) -> Annotated[str, "File path of the edited document."]:
    """Edit a document by inserting text at specific line numbers."""
    folder = session_folder if session_folder else "research_sessions/final"
    file_path = os.path.join(folder, file_name)
    
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Process insertions in order
    sorted_inserts = sorted(inserts.items())

    # Insert text at specified line numbers
    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    # Save edited document to file
    with open(file_path, "w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"

# Initialize LLM
MODEL_NAME = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)

# Agent Factory Class
class AgentFactory:
    def __init__(self, model_name):
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def create_agent_node(self, agent, name: str):
        # Node creation function
        def agent_node(state):
            # Add agent name to the state for callback tracking
            state["agent_name"] = name
            result = agent.invoke(state)
            return {
                "messages": [
                    HumanMessage(content=result["messages"][-1].content, name=name)
                ],
                "agent_name": name  # Include agent name in the output
            }

        return agent_node

def create_team_supervisor(model_name, system_prompt, members) -> str:
    # Define list of options for next worker
    options_for_next = ["FINISH"] + members

    # Define response model for worker selection
    class RouteResponse(BaseModel):
        next: Literal[*options_for_next]

    # Create ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next? "
                "Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=str(options_for_next))

    # Initialize LLM
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Combine prompt and LLM to create chain
    supervisor_chain = prompt | llm.with_structured_output(RouteResponse)

    return supervisor_chain

# Define state
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

def get_last_message(state: State) -> dict:
    """Extract the last message and prepare it for the next agent."""
    last_message = state["messages"][-1]
    if isinstance(last_message, str):
        return {
            "messages": [HumanMessage(content=last_message)],
            "next": "Supervisor",
            "agent_name": None
        }
    else:
        return {
            "messages": [last_message],
            "next": "Supervisor",
            "agent_name": None
        }

def join_graph(response: dict) -> dict:
    """Consolidate responses from sub-graphs."""
    return {
        "messages": [response["messages"][-1]],
        "next": "Supervisor",
        "agent_name": None
    }

def get_next_node(x: dict) -> str:
    """Determine the next node based on state."""
    return x.get("next", "FINISH")

# Custom callback handler for Streamlit
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler for displaying progress in Streamlit."""
    
    def __init__(self, container, agent_names=None):
        """Initialize the handler with a Streamlit container."""
        self.container = container
        self.step_count = 0
        self.current_agent = None
        
        # Create timestamped folder for this session
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = os.path.join("research_sessions", f"session_{timestamp}")
        os.makedirs(self.session_folder, exist_ok=True)
        
        # Initialize dynamic agent and phase tracking
        self.agent_info = {}  # Will store agent details from JSON responses
        self.phase_info = {}  # Will store phase information from JSON responses
        
        # Initialize session state for persistence
        if 'operations' not in st.session_state:
            st.session_state.operations = []
        
        if 'output_file' not in st.session_state:
            st.session_state.output_file = None
        
        # Create placeholders for progress
        with self.container:
            st.markdown("### 🚀 Research Progress")
            self.progress_placeholder = st.empty()
            self.current_operation_placeholder = st.empty()
            self.output_placeholder = st.empty()
            self._update_display()
    
    def _update_agent_info(self, data: Dict[str, Any]):
        """Update agent information from JSON response."""
        try:
            if isinstance(data, dict):
                # Extract agent information from messages
                if "messages" in data:
                    for msg in data["messages"]:
                        if hasattr(msg, 'name') and msg.name:
                            self.agent_info[msg.name] = {
                                'name': msg.name,
                                'type': getattr(msg, 'type', 'unknown'),
                                'last_seen': datetime.datetime.now()
                            }
                
                # Extract agent information from next field
                if "next" in data and data["next"]:
                    next_agent = data["next"]
                    if next_agent not in self.agent_info:
                        self.agent_info[next_agent] = {
                            'name': next_agent,
                            'type': 'agent',
                            'last_seen': datetime.datetime.now()
                        }
                
                # Extract phase information if available
                if "phase" in data:
                    phase = data["phase"]
                    if isinstance(phase, (str, int)) and phase not in self.phase_info:
                        self.phase_info[phase] = {
                            'name': str(phase),
                            'timestamp': datetime.datetime.now()
                        }
        except Exception as e:
            if st.session_state.get('debug_mode', False):
                st.error(f"Error updating agent info: {str(e)}")

    def _is_real_agent(self, agent_name: str) -> bool:
        """Check if the name represents a real agent rather than a tool or utility."""
        if not agent_name or agent_name == "None":
            return False
            
        real_agents = {
            "Supervisor", "Searcher", "WebScraper", 
            "DocWriter", "NoteTaker", "ChartGenerator",
            "ResearchTeam", "PaperWritingTeam",
            # Add variations that might appear in the workflow
            "Super Supervisor", "Web Searcher", "Web Scraper",
            "Document Writer", "Note Taker", "Chart Generator",
            "Research Team", "Paper Writing Team"
        }
        return agent_name in real_agents or any(agent in agent_name for agent in real_agents)

    def _get_display_name(self, agent_name: str) -> Optional[str]:
        """Get the display name for an agent."""
        if not agent_name or agent_name == "None":
            return None
            
        # Special handling for known agent names
        agent_display = {
            "Supervisor": "🎯 Super Supervisor",
            "Super Supervisor": "🎯 Super Supervisor",
            "Searcher": "🔍 Web Searcher",
            "Web Searcher": "🔍 Web Searcher",
            "WebScraper": "📑 Web Scraper",
            "Web Scraper": "📑 Web Scraper",
            "DocWriter": "✍️ Document Writer",
            "Document Writer": "✍️ Document Writer",
            "NoteTaker": "📋 Note Taker",
            "Note Taker": "📋 Note Taker",
            "ChartGenerator": "📊 Chart Generator",
            "Chart Generator": "📊 Chart Generator",
            "ResearchTeam": "🔬 Research Team",
            "Research Team": "🔬 Research Team",
            "PaperWritingTeam": "📝 Paper Writing Team",
            "Paper Writing Team": "📝 Paper Writing Team"
        }
        
        # Try exact match first
        if agent_name in agent_display:
            return agent_display[agent_name]
            
        # Try partial match if exact match fails
        for key, value in agent_display.items():
            if key.lower() in agent_name.lower():
                return value
                
        return None

    def _get_phase_name(self, step_number: int) -> str:
        """Get the phase name from step number."""
        if step_number in self.phase_info:
            return self.phase_info[step_number]['name']
        return f"Phase {step_number}"

    def _get_agent_name(self, serialized: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        """Extract agent name from various possible locations."""
        try:
            # Check if serialized is None
            if serialized is None:
                if isinstance(inputs, dict):
                    # Try to get from agent_name
                    if 'agent_name' in inputs:
                        return inputs['agent_name']
                    # Try to get from next field
                    if 'next' in inputs:
                        return inputs['next']
                    # Try to get from messages
                    if 'messages' in inputs and inputs['messages']:
                        msg = inputs['messages'][-1]
                        if hasattr(msg, 'name'):
                            return msg.name
                return "UnknownAgent"
            
            if isinstance(serialized, dict):
                # Try different locations for the agent name
                agent_name = (
                    serialized.get('name') or 
                    (serialized.get('config', {}) or {}).get('name') or
                    serialized.get('agent_name') or
                    serialized.get('next')  # Also check 'next' field in serialized
                )
                if agent_name:
                    return agent_name
                
                # Try to get from class name
                class_name = serialized.get('class_name', '')
                if class_name:
                    # Map class names to agent names
                    class_to_agent = {
                        'DocWriter': 'DocWriter',
                        'NoteTaker': 'NoteTaker',
                        'ChartGenerator': 'ChartGenerator',
                        'Searcher': 'Searcher',
                        'WebScraper': 'WebScraper',
                        'Supervisor': 'Supervisor',
                        'ResearchTeam': 'ResearchTeam',
                        'PaperWritingTeam': 'PaperWritingTeam'
                    }
                    for key, value in class_to_agent.items():
                        if key in class_name:
                            return value
            
            return "UnknownAgent"
            
        except Exception as e:
            st.error(f"Error in _get_agent_name: {str(e)}")
            return "UnknownAgent"

    def _format_data_for_log(self, data: Any) -> str:
        """Format data for log file in a readable way."""
        try:
            return str(data)
        except Exception as e:
            return f"Error formatting data: {str(e)}"

    def _update_display(self):
        """Update the display of operations."""
        with self.container:
            if st.session_state.operations:
                # Show current step status
                if self.current_agent and isinstance(self.current_agent, str):
                    display_name = self._get_display_name(self.current_agent)
                    if not display_name:  # Skip if no valid display name
                        return None
                    if "None" in display_name:
                        return None
                    if display_name:  # Only show if we have a valid display name
                        st.info(f"🔄 {display_name} is working...")
                
                # Show progress list
                st.markdown("#### Research Progress:")
                progress_md = ""
                
                # Save detailed operations to file
                details_file = os.path.join(self.session_folder, "operation_details.log")
                try:
                    with open(details_file, 'w', encoding='utf-8') as f:
                        for op in st.session_state.operations:
                            _, timestamp, agent, status, message = op
                            
                            # Skip if agent is not valid
                            if not agent or not isinstance(agent, str):
                                continue
                            
                            display_name = self._get_display_name(agent)
                            if not display_name:  # Skip if no valid display name
                                continue
                            
                            if "None" in display_name:
                                continue
                            
                            # Write details to file with emoji-safe handling
                            try:
                                f.write(f"\n{'='*50}\n")
                                f.write(f"[{timestamp}] {display_name} - {status}\n")
                                if message:
                                    f.write(f"Details:\n{message}\n")
                                f.write(f"{'='*50}\n")
                                
                                # Update progress display only for valid agents
                                icon = "✅" if status == "✅ Completed" else "🔄"
                                progress_md += f"{icon} {display_name}\n"
                            except UnicodeEncodeError:
                                # Fallback without emojis if encoding fails
                                clean_display_name = ''.join(c for c in display_name if ord(c) < 0x10000)
                                f.write(f"[{timestamp}] {clean_display_name} - {status}\n")
                                if message:
                                    f.write(f"Details:\n{message}\n")
                                f.write("-" * 50 + "\n")
                except Exception as e:
                    st.error(f"Error writing to log file: {str(e)}")

            # Display all research sessions
            self._list_research_sessions()

    def _serialize_message(self, obj):
        """Serialize message objects and other types for JSON."""
        if hasattr(obj, 'content') and hasattr(obj, 'type'):
            # Handle LangChain message types
            return {
                'type': obj.type,
                'content': obj.content,
                'additional_kwargs': obj.additional_kwargs,
                'name': getattr(obj, 'name', None)
            }
        elif hasattr(obj, '__dict__'):
            # Handle other objects with attributes
            return obj.__dict__
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            # Handle datetime objects
            return obj.isoformat()
        else:
            # Handle basic types
            try:
                return str(obj)
            except:
                return None

    def _save_llm_response(self, response_data: Dict[str, Any], prefix: str = "llm"):
        """Save LLM response as JSON file."""
        try:
            self.response_counter += 1
            timestamp = datetime.datetime.now().strftime("%H%M%S")
            filename = f"{prefix}_{timestamp}_{self.response_counter}.json"
            filepath = os.path.join(self.session_folder, filename)
            
            # Update agent and phase information
            self._update_agent_info(response_data)
            
            # Convert response data to serializable format
            def serialize_data(data):
                if isinstance(data, dict):
                    return {k: serialize_data(v) for k, v in data.items()}
                elif isinstance(data, list):
                    return [serialize_data(item) for item in data]
                else:
                    return self._serialize_message(data)
            
            serialized_data = serialize_data(response_data)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False, default=self._serialize_message)
            
            return filepath
        except Exception as e:
            st.error(f"Error saving LLM response: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.error(f"Response data: {response_data}")
            return None

    def _add_operation(self, agent_name: str, status: str, message: str = None):
        """Add a new operation to the sequence."""
        try:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Ensure message is properly formatted
            if message and not isinstance(message, str):
                try:
                    if hasattr(message, 'content'):
                        message = message.content
                    else:
                        message = str(message)
                except:
                    message = "Message conversion failed"
            
            operation = (self.step_count, timestamp, agent_name, status, message)
            st.session_state.operations.append(operation)
            self._update_display()
        except Exception as e:
            st.error(f"Error adding operation: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.error(f"Operation details: {agent_name=}, {status=}, {message=}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Handle chain start."""
        try:
            agent_name = self._get_agent_name(serialized, inputs)
            if agent_name:
                self.current_agent = agent_name
                self.step_count += 1
                
                # Add operation with raw input
                self._add_operation(
                    agent_name,
                    "🔄 Started",
                    f"Input:\n{str(inputs)}"
                )
                
                # Update current status
                display_name = self._get_display_name(agent_name)
                if not display_name:  # Skip if no valid display name
                    return None
                if "None" in display_name:
                    return None
                if display_name:  # Only show if we have a valid display name
                    st.info(f"🔄 {display_name} is working...")
        except Exception as e:
            st.error(f"Error in on_chain_start: {str(e)}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Handle chain end."""
        try:
            if self.current_agent:
                # Save operation to log file without displaying
                log_entry = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "agent": self.current_agent,
                    "status": "completed",
                    "output": str(outputs)
                }
                
                # Write to log file
                log_file = os.path.join(self.session_folder, "operation_details.log")
                try:
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(f"\n{'='*50}\n")
                        f.write(f"[{log_entry['timestamp']}] {self.current_agent}\n")
                        f.write(f"Status: {log_entry['status']}\n")
                        f.write(f"Output:\n{log_entry['output']}\n")
                        f.write(f"{'='*50}\n")
                except Exception as e:
                    st.error(f"Error writing to log file: {str(e)}")
                
                # Save output to MD file if it's from DocWriter or final output
                if self.current_agent in ["DocWriter", "PaperWritingTeam"]:
                    try:
                        content = None
                        if hasattr(outputs, 'content'):
                            content = outputs.content
                        elif isinstance(outputs, dict) and "messages" in outputs:
                            messages = outputs["messages"]
                            if messages and hasattr(messages[-1], 'content'):
                                content = messages[-1].content
                        elif isinstance(outputs, str):
                            content = outputs
                        
                        if content:
                            self.save_output_to_md(content)
                    except Exception as e:
                        st.error(f"Error saving markdown output: {str(e)}")
                
                # Update display name for status
                display_name = self._get_display_name(self.current_agent)
                if display_name and "None" not in display_name:
                    st.success(f"✅ {display_name} completed")
                
                self.current_agent = None
                
        except Exception as e:
            st.error(f"Error in on_chain_end: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.error(f"Outputs type: {type(outputs)}")

    def save_output_to_md(self, content: str, filename: str = None):
        """Save the research output to a markdown file."""
        try:
            if filename is None:
                # Generate filename based on first line of content or timestamp
                first_line = content.split('\n')[0][:50]  # Take first 50 chars of first line
                safe_name = re.sub(r'[^\w\s-]', '', first_line).strip().replace(' ', '_')
                if not safe_name:
                    safe_name = "research_output"
                filename = safe_name
            
            filepath = os.path.join(self.session_folder, filename)
            
            # Save the content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Also save a copy as final_output
            final_output_path = os.path.join(self.session_folder, "final_output")
            with open(final_output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            st.session_state.output_file = filepath
            
        except Exception as e:
            st.error(f"Error saving output file: {str(e)}")
            if st.session_state.get('debug_mode', False):
                st.error(f"Attempted to save to: {filepath}")
    
    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        """Handle chain errors."""
        if self.current_agent:
            self._add_operation(
                self.current_agent,
                "❌ Error",
                f"Error: {str(error)}"
            )
    
    def reset(self):
        """Reset the handler state."""
        st.session_state.operations = []
        st.session_state.output_file = None
        self.step_count = 0
        self.current_agent = None
        self._update_display()

    def _list_research_sessions(self):
        """List all research sessions and their contents."""
        if os.path.exists("research_sessions"):
            sessions = sorted([d for d in os.listdir("research_sessions") 
                             if os.path.isdir(os.path.join("research_sessions", d))],
                            reverse=True)  # Most recent first
            
            if sessions:
                st.markdown("### 📂 Research Sessions")
                for session in sessions:
                    session_path = os.path.join("research_sessions", session)
                    # Get session timestamp from folder name
                    timestamp = session.replace("session_", "")
                    try:
                        # Convert timestamp to readable format
                        dt = datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                        readable_time = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        readable_time = timestamp
                    
                    st.markdown(f"#### 📁 Session: {readable_time}")
                    
                    # Group files by type
                    md_files = []
                    other_files = []
                    
                    for root, _, files in os.walk(session_path):
                        for file in sorted(files):
                            file_path = os.path.join(root, file)
                            rel_path = os.path.relpath(file_path, session_path)
                            mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
                            file_time = mod_time.strftime("%H:%M:%S")
                            
                            if file.endswith('.md'):
                                md_files.append((rel_path, file_path, file_time))
                            else:
                                other_files.append((rel_path, file_path, file_time))
                    
                    # Display markdown files
                    if md_files:
                        st.markdown("##### 📝 Documents")
                        for rel_path, file_path, file_time in md_files:
                            st.markdown(f"📄 [{file_time}] {rel_path}")
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    st.markdown(f.read())
                            except Exception as e:
                                st.error(f"Error reading {rel_path}: {str(e)}")
                            st.markdown("---")
                    
                    # Display other files
                    if other_files:
                        st.markdown("##### 📎 Other Files")
                        for rel_path, file_path, file_time in other_files:
                            st.markdown(f"📎 [{file_time}] {rel_path}")
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    st.text(f.read())
                            except Exception as e:
                                st.error(f"Error reading {rel_path}: {str(e)}")
                            st.markdown("---")
                    
                    st.markdown("---")
            else:
                st.info("No research sessions found.")
        else:
            st.info("Research sessions directory not found.")

def run_graph(graph, query: str, recursive_limit: int = 25) -> dict:
    """Helper function to run the graph and stream tokens"""
    config = {
        "configurable": {
            "thread_id": "1",
        }
    }
    
    final_response = ""
    for s in graph.stream(
        {
            "messages": [
                HumanMessage(
                    content=query
                )
            ],
        },
        config,
        stream_mode="values",
    ):
        if "messages" in s:
            final_response = s["messages"][-1].content
    return final_response

# Streamlit UI
st.markdown("<h1>🔬 Research Assistant</h1>", unsafe_allow_html=True)

# Create a container for the introduction
with st.container():
    st.markdown("""
    <div class="info">
        Welcome to the Research Assistant! This AI-powered tool helps you conduct thorough research 
        and generate comprehensive papers on any topic. Choose your research type below and let our 
        team of specialized agents help you create high-quality content.
    </div>
    """, unsafe_allow_html=True)

# Create columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Query type selection with custom styling
    st.markdown("### 📝 Select Your Research Type")
    query_type = st.radio(
        "",  # Empty label as we're using markdown above
        ["Simple Query", "Complex Research Paper"],
        help="Choose 'Simple Query' for quick research or 'Complex Research Paper' for detailed academic papers"
    )

    # Query input with dynamic placeholder and height
    st.markdown("### 🎯 Enter Your Topic")
    if query_type == "Simple Query":
        query = st.text_area(
            "",
            height=100,
            placeholder="Enter your research query here...\nExample: What are the latest developments in quantum computing?",
            help="Enter a specific question or topic you'd like to research"
        )
        recursive_limit = st.slider(
            "🔄 Processing Depth",
            min_value=10,
            max_value=30,
            value=15,
            help="Higher values allow for more complex queries but take longer to process"
        )
    else:
        query = st.text_area(
            "",
            height=100,
            placeholder="Enter your research paper topic here...\nExample: The Impact of Artificial Intelligence on Modern Healthcare",
            help="Enter the main topic for your research paper"
        )
        # For research paper, just pass the topic, not the full paper prompt
        if query:
            query = f"Topic: {query}"
        recursive_limit = st.slider(
            "🔄 Research Depth",
            min_value=50,
            max_value=200,
            value=100,
            help="Higher values allow for more detailed research but take longer to process"
        )

with col2:
    # Team information in a card
    st.markdown("""
    <div class="team-card">
        <h3>🤖 Our Research Team</h3>
        <p><span class="agent-icon">🔍</span> <strong>Searcher:</strong> Finding information</p>
        <p><span class="agent-icon">📑</span> <strong>WebScraper:</strong> Extracting content</p>
        <p><span class="agent-icon">✍️</span> <strong>DocWriter:</strong> Writing papers</p>
        <p><span class="agent-icon">📋</span> <strong>NoteTaker:</strong> Creating outlines</p>
        <p><span class="agent-icon">📊</span> <strong>ChartGenerator:</strong> Creating visuals</p>
        <p><span class="agent-icon">🎯</span> <strong>Supervisors:</strong> Coordinating work</p>
    </div>
    """, unsafe_allow_html=True)

# Start Research button with custom styling
if st.button("🚀 Start Research", help="Click to begin the research process"):
    if query:
        try:
            # Create a progress container with custom styling
            progress_container = st.empty()
            with progress_container.container():
                st.markdown("""
                <div class="stCard">
                    <h3>🔄 Research Progress</h3>
                    <div class="progress-content"></div>
                </div>
                """, unsafe_allow_html=True)

            # Show team information
            st.markdown("""
            ### 🚀 Active Research Teams

            #### 🔬 Research Team
            - 🔍 Searcher: Finding relevant information
            - 📑 WebScraper: Extracting detailed content
            - 🎯 Research Supervisor: Coordinating research

            #### 📝 Paper Writing Team
            - ✍️ DocWriter: Writing the paper
            - 📋 NoteTaker: Creating outlines and summaries
            - 📊 ChartGenerator: Creating visualizations
            - 🎭 Writing Supervisor: Coordinating writing

            #### 🎭 Super Team
            - 🔬 ResearchTeam: Gathering information
            - 📝 PaperWritingTeam: Creating the paper
            - 🎭 Super Supervisor: Orchestrating teams
            """)

            # Rest of your existing code for agent initialization and graph creation
            python_repl_tool = PythonREPLTool()

            # Initialize agents and graph
            agent_factory = AgentFactory(MODEL_NAME)
            
            # Create search node
            search_agent = create_react_agent(
                llm, 
                tools=[tavily_tool]
            )
            search_node = agent_factory.create_agent_node(search_agent, name="Searcher")

            # Create web scraping node
            web_scraping_agent = create_react_agent(
                llm, 
                tools=[scrape_webpages]
            )
            web_scraping_node = agent_factory.create_agent_node(web_scraping_agent, name="WebScraper")

            # Create document writing agent
            doc_writer_agent = create_react_agent(
                llm,
                tools=[write_document, edit_document, read_document],
                state_modifier="""
You are an expert research paper writer. For each section and subsection, write several detailed paragraphs (not just a summary or bullet points). Each subsection should include explanations, examples, and references where possible. Expand on every point in depth, aiming for comprehensive, multi-paragraph content.
When you use the write_document tool, always set session_folder to 'research_sessions/final' so the file is saved in the correct folder.
""",
            )
            context_aware_doc_writer_agent = get_last_message | doc_writer_agent
            doc_writing_node = agent_factory.create_agent_node(
                context_aware_doc_writer_agent, name="DocWriter"
            )

            # Create note taking node
            note_taking_agent = create_react_agent(
                llm,
                tools=[create_outline, read_document],
                state_modifier="""
You are an expert in creating outlines for research papers. Your mission is to create an outline for a given topic/resources or documents.
When you use the create_outline tool, always set session_folder to 'research_sessions/final' so the file is saved in the correct folder.
""",
            )
            context_aware_note_taking_agent = get_last_message | note_taking_agent
            note_taking_node = agent_factory.create_agent_node(
                context_aware_note_taking_agent, name="NoteTaker"
            )

            # Create chart generating agent
            chart_generating_agent = create_react_agent(
                llm,
                tools=[read_document, python_repl_tool]
            )
            context_aware_chart_generating_agent = get_last_message | chart_generating_agent
            chart_generating_node = agent_factory.create_agent_node(
                context_aware_chart_generating_agent, name="ChartGenerator"
            )

            # Create Supervisor for research team
            research_supervisor_prompt = """"You are a supervisor tasked with managing a conversation between the"
    " following workers: Search, WebScraper. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."

Current conversation:
{messages}

Who should act next? Select one: {options}"""
            
            research_supervisor = create_team_supervisor(
                MODEL_NAME,
                research_supervisor_prompt,
                ["Searcher", "WebScraper"]
            )

            # Create Supervisor for paper writing team
            writing_supervisor_prompt = """
You are a supervisor tasked with managing a conversation between the following workers: ['DocWriter', 'NoteTaker', 'ChartGenerator'].
For every research paper, you must:
- First, use the NoteTaker to create a detailed outline and save it as a file in the /final folder.
- Only after the outline is created, use the DocWriter to write the full research paper in arXiv format, in .md (Markdown) format, with clear sections and subsections, and save it in the /final folder. The paper must be at least 2 pages long (minimum 2000 words recommended), with multiple paragraphs per section. The DocWriter must include citations throughout the paper, using academic papers, articles, and reliable sources in APA format, and end with a References section.
- Use the ChartGenerator if any visuals or charts are needed.
- When using write_document or create_outline, always set session_folder to 'research_sessions/final'.
Do not finish until all steps are complete and the final .md file is written and saved in the /final folder.
Given the following user request, respond with the worker to act next. Each worker will perform a task and respond with their results and status. When finished, respond with FINISH.

Current conversation:
{messages}

Who should act next? Select one: {options}"""
            
            writing_supervisor = create_team_supervisor(
                MODEL_NAME,
                writing_supervisor_prompt,
                ["DocWriter", "NoteTaker", "ChartGenerator"]
            )

            # Create Super Supervisor
            super_supervisor_prompt = """"You are a supervisor tasked with managing a conversation between the"
    " following teams: ['ResearchTeam', 'PaperWritingTeam']. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH.

Current conversation:
{messages}

Who should act next? Select one: {options}"""
            
            super_supervisor = create_team_supervisor(
                MODEL_NAME,
                super_supervisor_prompt,
                ["ResearchTeam", "PaperWritingTeam"]
            )

            # Create research graph
            web_research_graph = StateGraph(ResearchState)
            web_research_graph.add_node("Searcher", search_node)
            web_research_graph.add_node("WebScraper", web_scraping_node)
            web_research_graph.add_node("Supervisor", research_supervisor)
            web_research_graph.add_edge("Searcher", "Supervisor")
            web_research_graph.add_edge("WebScraper", "Supervisor")
            web_research_graph.add_conditional_edges(
                "Supervisor",
                get_next_node,
                {"Searcher": "Searcher", "WebScraper": "WebScraper", "FINISH": END}
            )
            web_research_graph.set_entry_point("Supervisor")
            web_research_app = web_research_graph.compile()

            # Create authoring graph
            authoring_graph = StateGraph(DocWritingState)
            authoring_graph.add_node("DocWriter", doc_writing_node)
            authoring_graph.add_node("NoteTaker", note_taking_node)
            authoring_graph.add_node("ChartGenerator", chart_generating_node)
            authoring_graph.add_node("Supervisor", writing_supervisor)
            authoring_graph.add_edge("DocWriter", "Supervisor")
            authoring_graph.add_edge("NoteTaker", "Supervisor")
            authoring_graph.add_edge("ChartGenerator", "Supervisor")
            authoring_graph.add_conditional_edges(
                "Supervisor",
                get_next_node,
                {
                    "DocWriter": "DocWriter",
                    "NoteTaker": "NoteTaker",
                    "ChartGenerator": "ChartGenerator",
                    "FINISH": END
                }
            )
            authoring_graph.set_entry_point("Supervisor")
            authoring_app = authoring_graph.compile()

            # Create super graph
            super_graph = StateGraph(State)
            super_graph.add_node("ResearchTeam", get_last_message | web_research_app | join_graph)
            super_graph.add_node("PaperWritingTeam", get_last_message | authoring_app | join_graph)
            super_graph.add_node("Supervisor", super_supervisor)
            super_graph.add_edge("ResearchTeam", "Supervisor")
            super_graph.add_edge("PaperWritingTeam", "Supervisor")
            super_graph.add_conditional_edges(
                "Supervisor",
                get_next_node,
                {
                    "ResearchTeam": "ResearchTeam",
                    "PaperWritingTeam": "PaperWritingTeam",
                    "FINISH": END
                }
            )
            super_graph.set_entry_point("Supervisor")
            super_graph = super_graph.compile()

            # Run the graph with progress tracking
            result = run_graph(super_graph, query, recursive_limit)
                
            if result:
                # Display final summary
                st.markdown("### 📝 Final Results")
                if "messages" in result and result["messages"]:
                    final_message = result["messages"][-1].content
                    st.markdown(final_message)
                else:
                    st.error("No results were generated. Please try again with a higher recursion limit.")
        except Exception as e:
            st.markdown(f"""
            <div class="error">
                <h4>❌ Error Occurred</h4>
                <p>{str(e)}</p>
                <pre>{traceback.format_exc()}</pre>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="error">
            <h4>⚠️ Input Required</h4>
            <p>Please enter a query before starting the research process.</p>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    st.title("AI Research Assistant")
    st.sidebar.header("Controls")
    
    query = st.text_input("Enter your research topic:", "The impact of AI on scientific research")
    
    # Session state for managing chat history and run ID
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "run_id" not in st.session_state:
        st.session_state.run_id = str(uuid.uuid4())

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.sidebar.button("Start Research"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Initialize callback handler
        progress_container = st.empty()
        agent_names = {
            "researcher": "Researcher",
            "editor": "Editor",
            "doc_writer": "Document Writer",
            "supervisor": "Supervisor"
        }
        handler = StreamlitCallbackHandler(progress_container, agent_names)
        
        # Create agent instances
        agent_factory = AgentFactory(MODEL_NAME)

        # Research Agent
        research_agent = create_react_agent(
            llm,
            tools=[tavily_tool, scrape_webpages],
            messages_modifier=RESEARCHER_PROMPT,
        )
        research_node = agent_factory.create_agent_node(research_agent, "researcher")

        # Editor Agent
        editor_agent = create_react_agent(
            llm,
            tools=[read_document, edit_document],
            messages_modifier=EDITOR_PROMPT,
        )
        editor_node = agent_factory.create_agent_node(editor_agent, "editor")

        # Document Writing Agent
        doc_writer_agent = create_react_agent(
            llm,
            tools=[create_outline, write_document, read_document],
            messages_modifier=DOC_WRITER_PROMPT,
        )
        doc_writer_node = agent_factory.create_agent_node(doc_writer_agent, "doc_writer")

        # Supervisor Agent
        supervisor_agent = create_team_supervisor(
            MODEL_NAME,
            RESEARCH_SUPERVISOR_PROMPT.format(members="researcher, editor, doc_writer", query=query),
            ["researcher", "editor", "doc_writer"]
        )

        # Define the graph
        workflow = StateGraph(ResearchState)
        workflow.add_node("researcher", research_node)
        workflow.add_node("editor", editor_node)
        workflow.add_node("doc_writer", doc_writer_node)
        workflow.add_node("supervisor", supervisor_agent)

        # Define edges
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("editor", "supervisor")
        workflow.add_edge("doc_writer", "supervisor")
        
        workflow.add_conditional_edges(
            "supervisor",
            lambda x: x["next"],
            {
                "researcher": "researcher",
                "editor": "editor",
                "doc_writer": "doc_writer",
                "FINISH": END,
            },
        )
        workflow.set_entry_point("supervisor")
        
        # Compile graph
        graph = workflow.compile(checkpointer=MemorySaver())
        
        # Run the graph
        with st.spinner("Research in progress..."):
            final_response = run_graph(graph, query)
            st.session_state.messages.append({"role": "assistant", "content": final_response})
            with st.chat_message("assistant"):
                st.markdown(final_response)
            handler.save_output_to_md(final_response)

if __name__ == "__main__":
    main()