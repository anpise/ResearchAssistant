from langchain_openai.chat_models import ChatOpenAI

# Initialize LLM
MODEL_NAME = "gpt-4o-mini"
llm = ChatOpenAI(model=MODEL_NAME, temperature=0) 