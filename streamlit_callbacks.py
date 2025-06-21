from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any, Union
import streamlit as st
import datetime
import json
import os

class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container, agent_names=None):
        super().__init__()
        self.container = container
        self.agent_names = agent_names if agent_names else {}
        self.step_number = 0
        self.agent_info = {}
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"research_sessions/{self.session_id}"
        os.makedirs(self.session_folder, exist_ok=True)
        self.llm_response_folder = os.path.join(self.session_folder, "llm_responses")
        os.makedirs(self.llm_response_folder, exist_ok=True)

    def _update_agent_info(self, data: Dict[str, Any]):
        agent_name = data.get("name")
        if agent_name and self._is_real_agent(agent_name):
            if agent_name not in self.agent_info:
                display_name = self._get_display_name(agent_name)
                self.agent_info[agent_name] = {
                    "name": display_name,
                    "color": "#00E5FF",  # Assign a color
                    "operations": [],
                    "status": "pending"
                }

    def _is_real_agent(self, agent_name: str) -> bool:
        return agent_name in self.agent_names

    def _get_display_name(self, agent_name: str) -> Optional[str]:
        return self.agent_names.get(agent_name)

    def _get_phase_name(self, step_number: int) -> str:
        if 1 <= step_number <= 2:
            return "Research & Analysis"
        elif step_number == 3:
            return "Content Generation"
        elif step_number == 4:
            return "Review & Refine"
        return "Finalizing"

    def _get_agent_name(self, serialized: Dict[str, Any], inputs: Dict[str, Any]) -> str:
        # Check for agent name in various possible locations
        if "agent_name" in inputs:
            return inputs["agent_name"]
        if "name" in serialized:
            return serialized["name"]
        # Fallback if direct properties are not found
        if "graph" in serialized and "nodes" in serialized["graph"]:
            for node in serialized["graph"]["nodes"]:
                if "attrs" in node and "title" in node["attrs"]:
                    return node["attrs"]["title"]
        return "Unknown Agent"

    def _format_data_for_log(self, data: Any) -> str:
        if isinstance(data, dict):
            return json.dumps(data, indent=2, default=self._serialize_message)
        return str(data)

    def _update_display(self):
        self.container.empty()
        with self.container:
            st.header(f"Research Progress (Session: {self.session_id})")
            
            phase_name = self._get_phase_name(self.step_number)
            st.subheader(f"Phase: {phase_name}")
            
            cols = st.columns(len(self.agent_info))
            agent_keys = list(self.agent_info.keys())

            for i, col in enumerate(cols):
                if i < len(agent_keys):
                    agent_name = agent_keys[i]
                    agent_data = self.agent_info[agent_name]
                    with col:
                        st.markdown(f"""
                        <div style="
                            border: 2px solid {agent_data['color']}; 
                            border-radius: 8px; 
                            padding: 10px; 
                            background-color: #222;
                            margin-bottom: 10px;
                        ">
                            <h4 style="color: {agent_data['color']};">{agent_data['name']}</h4>
                            <p><strong>Status:</strong> {agent_data['status']}</p>
                        </div>
                        """, unsafe_allow_html=True)

            st.subheader("Operation Log")
            log_content = ""
            for agent_name, agent_data in self.agent_info.items():
                if agent_data["operations"]:
                    log_content += f"**{agent_data['name']}**:\n"
                    for op in agent_data["operations"]:
                        log_content += f"- {op}\n"
            st.markdown(log_content)

    def _serialize_message(self, obj):
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if hasattr(obj, "to_json"):
            return obj.to_json()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def _save_llm_response(self, response_data: Dict[str, Any], prefix: str = "llm"):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.json"
        
        # Ensure nested objects are serializable
        def serialize_data(data):
            if isinstance(data, dict):
                return {k: serialize_data(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [serialize_data(i) for i in data]
            else:
                return self._serialize_message(data)

        try:
            with open(os.path.join(self.llm_response_folder, filename), 'w') as f:
                json.dump(serialize_data(response_data), f, indent=4)
        except Exception as e:
            st.error(f"Failed to save LLM response: {e}")

    def _add_operation(self, agent_name: str, status: str, message: str = None):
        if agent_name in self.agent_info:
            op_message = f"{status}"
            if message:
                op_message += f": {message}"
            self.agent_info[agent_name]["operations"].append(op_message)
            self.agent_info[agent_name]["status"] = status
            self._update_display()

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        agent_name = self._get_agent_name(serialized, inputs)
        
        if self._is_real_agent(agent_name):
            self.step_number += 1
            if agent_name not in self.agent_info:
                self._update_agent_info({"name": agent_name})

            self._add_operation(agent_name, "In Progress")
            
            # Log inputs
            log_data = {
                "step": self.step_number,
                "agent": self._get_display_name(agent_name),
                "inputs": self._format_data_for_log(inputs),
            }
            self._save_llm_response(log_data, prefix=f"start_{agent_name}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        if "agent_name" in outputs:
            agent_name = outputs["agent_name"]
            if self._is_real_agent(agent_name):
                self._add_operation(agent_name, "Completed")
                
                # Log outputs
                log_data = {
                    "step": self.step_number,
                    "agent": self._get_display_name(agent_name),
                    "outputs": self._format_data_for_log(outputs),
                }
                self._save_llm_response(log_data, prefix=f"end_{agent_name}")

    def save_output_to_md(self, content: str, filename: str = None):
        """Saves the final output to a Markdown file."""
        if filename is None:
            # Generate a filename if not provided
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"research_report_{timestamp}.md"
        
        # Ensure the 'results' directory exists
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        file_path = os.path.join(results_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        st.success(f"Output saved to {file_path}")
        return file_path

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any) -> None:
        agent_name = kwargs.get("agent_name", "Unknown Agent")
        if self._is_real_agent(agent_name):
            self._add_operation(agent_name, "Error", str(error))
        st.error(f"An error occurred in {agent_name}: {error}")

    def reset(self):
        self.step_number = 0
        self.agent_info = {}
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_folder = f"research_sessions/{self.session_id}"
        os.makedirs(self.session_folder, exist_ok=True)
        self.llm_response_folder = os.path.join(self.session_folder, "llm_responses")
        os.makedirs(self.llm_response_folder, exist_ok=True)

    def _list_research_sessions(self):
        base_dir = "research_sessions"
        if not os.path.exists(base_dir):
            return []
        
        session_folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        
        session_data = []
        for session in session_folders:
            session_path = os.path.join(base_dir, session)
            report_file = self._find_report_file(session_path)
            session_data.append({
                "id": session,
                "path": session_path,
                "report_file": report_file
            })
            
        return sorted(session_data, key=lambda x: x['id'], reverse=True)

    def _find_report_file(self, session_path):
        final_dir = os.path.join(session_path, "final")
        if not os.path.exists(final_dir):
            return None
            
        for file in os.listdir(final_dir):
            if file.endswith(".md"):
                return os.path.join(final_dir, file)
        return None 