from agent import create_agent, process_query

def run_examples():
    """Run the required examples and display the transcripts."""
    print("Initializing the agent...")
    agent = create_agent()
    print("Agent initialized!\n")
    
    # Define the example queries
    counting_examples = [
        "How many 'r' characters are in 'strawberry'?",
        "Count the number of vowels in 'beautiful'",
        "How many times does the substring 'is' appear in 'This is a simple test to see if this is working'?"
    ]
    
    arithmetic_examples = [
        "What is 1234 * 5678?",
        "Calculate the sum of all numbers from 1 to 100"
    ]
    
    # Run counting examples
    print("=" * 50)
    print("COUNTING EXAMPLES")
    print("=" * 50)
    
    for i, query in enumerate(counting_examples, 1):
        print(f"\nCounting Example {i}:")
        result = process_query(agent, query)
        
        print(f"User: {result['user_input']}")
        print(f"Tool: {result['tool_name']}")
        print(f"Tool Input: {result['tool_input']}")
        print(f"Tool Output: {result['tool_output']}")
        print(f"Agent: {result['final_response']}")
        print("-" * 40)
    
    # Run arithmetic examples
    print("\n" + "=" * 50)
    print("ARITHMETIC EXAMPLES")
    print("=" * 50)
    
    for i, query in enumerate(arithmetic_examples, 1):
        print(f"\nArithmetic Example {i}:")
        result = process_query(agent, query)
        
        print(f"User: {result['user_input']}")
        print(f"Tool: {result['tool_name']}")
        print(f"Tool Input: {result['tool_input']}")
        print(f"Tool Output: {result['tool_output']}")
        print(f"Agent: {result['final_response']}")
        print("-" * 40)

if __name__ == "__main__":
    run_examples()
