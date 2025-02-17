# LangChain Conversational Agent

A lightweight yet powerful conversational AI agent built with LangChain and Claude 3. This agent can maintain conversations, perform web searches, and utilize various tools to assist users.

## Features

- 🤖 Powered by Anthropic's Claude 3 Sonnet
- 🔍 Web search capability via Tavily
- 💾 Conversation memory with ChromaDB
- 🛠️ Extensible tool system
- 📝 Structured response handling

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
export ANTHROPIC_API_KEY=your_api_key
export TAVILY_API_KEY=your_tavily_key
```

## Usage

Run the agent:
```bash
python project_main.py
```

## Project Structure

- `project_main.py`: Main agent implementation
- `requirements.txt`: Project dependencies
- `.gitignore`: Git ignore rules
- `vector_store/`: ChromaDB storage directory

## Requirements

- Python 3.8+
- Required packages are listed in requirements.txt 