# Tool-Calling Agent: Count Characters Correctly

This project demonstrates how to overcome tokenizer blindness in language models by connecting the model to Python execution capabilities. The agent can accurately perform counting and arithmetic operations by delegating these tasks to Python rather than relying on the model's internal representations.

## Overview

The agent uses LangChain with the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model and provides two tools:

1. **python.exec** - Executes Python code and returns stdout
2. **noop** - Returns nothing (used for normal, non-tool responses)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. The implementation uses the TinyLlama/TinyLlama-1.1B-Chat-v1.0 model, which will be downloaded automatically when you first run the code.

## Usage

### Interactive Mode

Run the agent in interactive mode:

```bash
python agent.py
```

This allows you to input queries and see the agent's responses in real-time.

### Example Transcripts

To see the predefined examples (3 counting queries and 2 arithmetic queries):

```bash
python examples.py
```

## Example Interactions

The agent can handle:

### Counting Queries
- "How many 'r' characters are in 'strawberry'?"
- "Count the number of vowels in 'beautiful'"
- "How many times does the substring 'is' appear in 'This is a simple test to see if this is working'?"

### Arithmetic Queries
- "What is 1234 * 5678?"
- "Calculate the sum of all numbers from 1 to 100"

## How It Works

1. The user submits a query
2. The agent decides whether to use Python execution or respond directly
3. If using Python, it generates appropriate code to solve the problem
4. The code is executed, and the result is returned
5. The agent formulates a human-friendly response based on the result

This approach ensures accurate results for tasks that require precise counting or arithmetic operations.
