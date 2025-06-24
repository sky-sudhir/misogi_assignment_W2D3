import sys
import io
import os
from typing import Dict, List, Tuple, Any

from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Import Google's Generative AI library
import google.generativeai as genai
from langchain_core.language_models.llms import LLM

# Define the tools

def python_exec(code: str) -> str:
    """Execute Python code and return the stdout output."""
    # Capture stdout
    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout
    
    try:
        # Execute the code
        exec(code, {})
        output = new_stdout.getvalue().strip()
        return output
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Restore stdout
        sys.stdout = old_stdout

def noop() -> str:
    """No operation tool that returns nothing."""
    return ""

# Create LangChain tools
tools = [
    Tool(
        name="python.exec",
        func=python_exec,
        description="Executes Python code and returns the stdout output. Use this for counting characters, arithmetic operations, or any computation."
    ),
    Tool(
        name="noop",
        func=noop,
        description="No operation. Use this for regular responses that don't require computation."
    )
]

# Create a custom LLM class for Gemini
class GeminiLLM(LLM):
    """LangChain wrapper for Google's Gemini API."""
    
    model_name: str = "gemini-2.0-flash"
    temperature: float = 0.1
    client: object = None
    
    def __init__(self, api_key=None, **kwargs):
        super().__init__(**kwargs)
        # Configure Gemini with API key
        api_key = api_key or os.environ.get("GOOGLE_API_KEY", "API_KEY_HERE")
        genai.configure(api_key=api_key)
        self.client = genai.GenerativeModel(self.model_name)
    
    def _call(self, prompt, stop=None):
        """Call the Gemini API with the prompt."""
        try:
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Error generating response."
    
    @property
    def _llm_type(self):
        """Return the type of LLM."""
        return "gemini"

# Initialize the Gemini model
def init_model():
    print("Initializing Gemini model...")
    llm = GeminiLLM(temperature=0.1)
    return llm

# Create the agent prompt
AGENT_PROMPT = """You are an intelligent assistant that can help with counting and arithmetic operations.

You have access to the following tools:

python.exec: Executes Python code and returns the stdout output. Use this for counting characters, arithmetic operations, or any computation.
noop: No operation. Use this for regular responses that don't require computation.

For tasks that require precise counting or arithmetic, use the python.exec tool with appropriate Python code.
For regular conversation, use the noop tool.

User Query: {input}

Think about whether you need to use a tool to answer this query accurately.

If you need to use the python.exec tool, format your response as:
<tool>python.exec</tool>
<tool_input>
# Python code here
</tool_input>

If you're providing a direct response without using a tool, format your response as:
<tool>noop</tool>
<tool_input></tool_input>

After the tool execution, you'll receive the result and should provide a final human-friendly response.
"""

def extract_tool_use(llm_output: str) -> Tuple[str, str]:
    """Extract tool name and input from LLM output."""
    tool_name = "noop"
    tool_input = ""
    
    if "<tool>" in llm_output and "</tool>" in llm_output:
        tool_part = llm_output.split("<tool>")[1].split("</tool>")[0].strip()
        tool_name = tool_part
    
    if "<tool_input>" in llm_output and "</tool_input>" in llm_output:
        tool_input_part = llm_output.split("<tool_input>")[1].split("</tool_input>")[0].strip()
        tool_input = tool_input_part
    
    return tool_name, tool_input

def use_tool(tool_name: str, tool_input: str) -> str:
    """Use the specified tool with the given input."""
    for tool in tools:
        if tool.name == tool_name:
            # Handle noop tool separately since it takes no arguments
            if tool_name == "noop":
                return tool.func()
            else:
                return tool.func(tool_input)
    return f"Error: Tool '{tool_name}' not found."

def format_conversation(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format the conversation history for the LLM."""
    formatted_messages = []
    for message in messages:
        if message["type"] == "human":
            formatted_messages.append(HumanMessage(content=message["content"]))
        elif message["type"] == "ai":
            formatted_messages.append(AIMessage(content=message["content"]))
    return formatted_messages

def create_agent():
    """Create the agent with the TinyLlama model."""
    llm = init_model()
    
    # Create the prompt
    prompt = ChatPromptTemplate.from_template(AGENT_PROMPT)
    
    # Create the chain
    chain = (
        {"input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def process_query(agent, query: str) -> Dict[str, str]:
    """Process a user query and return the full interaction details."""
    # Get the agent's initial response
    agent_response = agent.invoke(query)
    
    # Extract tool use
    tool_name, tool_input = extract_tool_use(agent_response)
    
    # Use the tool
    tool_output = use_tool(tool_name, tool_input)
    
    # Generate final response
    final_response = f"Based on the {tool_name} tool output: {tool_output}, the answer is: "
    if tool_name == "python.exec" and tool_output:
        final_response = f"The result is {tool_output}."
    else:
        final_response = agent_response.split("</tool_input>")[-1].strip()
        if not final_response:
            final_response = "I don't have a specific computation to perform for this query."
    
    # Return the full interaction
    return {
        "user_input": query,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "final_response": final_response
    }

def main():
    """Main function to run the agent."""
    print("Initializing the agent...")
    agent = create_agent()
    
    print("Agent ready! Enter 'exit' to quit.")
    
    while True:
        query = input("\nUser: ")
        if query.lower() == "exit":
            break
        
        result = process_query(agent, query)
        
        print("\nInteraction:")
        print(f"User: {result['user_input']}")
        if result['tool_name'] != "noop":
            print(f"Tool: {result['tool_name']}")
            print(f"Tool Input: {result['tool_input']}")
            print(f"Tool Output: {result['tool_output']}")
        print(f"Agent: {result['final_response']}")

if __name__ == "__main__":
    main()
