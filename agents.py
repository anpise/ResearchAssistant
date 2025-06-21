from langchain_openai.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage

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
        next: str

    # Create prompt template for supervisor
    supervisor_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",
            ),
        ]
    ).partial(options=options_for_next)
    
    # Create supervisor chain
    supervisor_chain = (
        supervisor_prompt
        | ChatOpenAI(model=model_name, temperature=0).with_structured_output(RouteResponse)
    )
    return supervisor_chain 