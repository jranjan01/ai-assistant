# AI Agent with RAG

A command-line chat agent that streams its answers, calls tools, and answers
questions from your own documents using retrieval-augmented generation (RAG).

### Project Setup and Execution

### 1. Install Dependencies

Install all required project dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Run the Project

Execute the Python application:

```bash
python ai_assistant.py
```

### 3. Linting and Formatting

This project uses Ruff for Python code linting and formatting.

**Check linting issues**

```bash
ruff check .
```

**Automatically fix linting issues**

```bash
ruff check . --fix
```

**Format Python code**

```bash
ruff format .
```

**Check code formatting**

```bash
ruff format --check .
```
