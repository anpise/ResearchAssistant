# AI Research Assistant

A Streamlit application that uses AI agents to perform comprehensive research, analysis, and documentation.

## Demo

Watch a video demonstration of the Research Assistant in action:

[Watch the Demo](Research%20Assistant%20Demo.mp4)

## Features

- **Researcher Agent**: Performs web searches and scrapes web pages to gather information.
- **Editor Agent**: Reviews and edits the research report for clarity, accuracy, and style.
- **Document Writer Agent**: Creates outlines and writes the research report.
- **Supervisor Agent**: Manages the workflow between the other agents.
- **Streamlit Interface**: An easy-to-use web interface to run the research assistant.
- **Integrated Tools**: Web search (Tavily) and web scraping capabilities.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

Run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

Then, open your web browser to the local URL provided by Streamlit. You can then enter your research topic and start the process.

## Project Structure
```
ResearchAssistant/
├── app.py                  # Main Streamlit application
├── agents.py               # Agent creation and management
├── prompts.py              # Prompts for the agents
├── tools.py                # Custom tools for the agents
├── config.py               # Configuration for the LLM
├── graph_state.py          # State definitions for the graph
├── streamlit_callbacks.py  # Callbacks for Streamlit UI
├── requirements.txt        # Project dependencies
├── results/                # Directory for saved research reports
└── README.md
```

## Project Documents

- [Final Presentation](annotated-PAS%20Final%20Presentation%20%281%29.pptx.pdf)
- [Project Report](PAS_Project_Report.pdf)
- [Research Results](results/) - Generated research reports and papers

## Contributing

Feel free to submit issues and enhancement requests! 