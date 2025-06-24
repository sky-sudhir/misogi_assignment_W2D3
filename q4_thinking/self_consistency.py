import json
import re
import argparse
from collections import Counter
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_problems(file_path):
    """Load problems from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_answer(text):
    """Extract numerical answer from model output."""
    # Look for patterns like "the answer is X" or "= X" or "X is the answer"
    patterns = [
        r"(?:answer|result)(?:\s+is|\s*=\s*|\s*:)\s*(-?\d+\.?\d*)",
        r"(?:=\s*)(-?\d+\.?\d*)",
        r"(-?\d+\.?\d*)(?:\s+is\s+the\s+(?:answer|result))",
        r"(?:final\s+(?:answer|result)(?:\s+is|\s*=\s*|\s*:)\s*)(-?\d+\.?\d*)",
        r"(?:Therefore,|Thus,|So,)(?:[^.]*?)(-?\d+\.?\d*)",
        r"(?:Therefore|Thus|So)(?:[^.]*?)(-?\d+\.?\d*)",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                # Return the last match as it's likely the final answer
                return float(matches[-1])
            except ValueError:
                continue
    
    # If no pattern matches, look for the last number in the text
    numbers = re.findall(r"(-?\d+\.?\d*)", text)
    if numbers:
        try:
            return float(numbers[-1])
        except ValueError:
            pass
    
    return None

def generate_completion(model, tokenizer, prompt, temperature=0):
    """Generate a completion using the model."""
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return completion

def format_prompt(problem):
    """Format the prompt for the model."""
    return f"""You are solving a math problem. Think step-by-step.

Problem: {problem}

Let's think step-by-step.
"""

def solve_problem_with_self_consistency(model, tokenizer, problem, num_samples=10, temperature=1.1):
    """Solve a problem using self-consistency approach."""
    prompt = format_prompt(problem)
    answers = []
    
    print(f"\nProblem: {problem}")
    print("\nGenerating multiple solutions...")
    
    for i in range(num_samples):
        completion = generate_completion(model, tokenizer, prompt, temperature)
        answer = extract_answer(completion)
        
        print(f"Sample {i+1} answer: {answer}")
        if answer is not None:
            answers.append(answer)
    
    if not answers:
        return None
    
    # Count occurrences of each answer
    answer_counts = Counter(answers)
    majority_answer, count = answer_counts.most_common(1)[0]
    
    print(f"\nAnswers distribution: {dict(answer_counts)}")
    print(f"Majority vote answer: {majority_answer} (appeared {count}/{len(answers)} times)")
    
    return majority_answer

def solve_problem_deterministic(model, tokenizer, problem):
    """Solve a problem using a single deterministic run."""
    prompt = format_prompt(problem)
    
    print(f"\nProblem: {problem}")
    print("Generating deterministic solution...")
    
    completion = generate_completion(model, tokenizer, prompt, temperature=0)
    answer = extract_answer(completion)
    
    print(f"Deterministic answer: {answer}")
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Thinking Mode Sampler")
    parser.add_argument("--problems_file", type=str, default="problems.json", help="Path to problems JSON file")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for self-consistency")
    parser.add_argument("--temperature", type=float, default=1.1, help="Temperature for sampling")
    args = parser.parse_args()
    
    # Load problems
    problems = load_problems(args.problems_file)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # Explicitly move model to CPU
    model = model.to("cpu")
    
    # Evaluate both methods
    deterministic_correct = 0
    self_consistency_correct = 0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    for problem_data in problems:
        problem_id = problem_data["id"]
        problem_text = problem_data["problem"]
        correct_answer = problem_data["answer"]
        
        print(f"\n\nProblem {problem_id}:")
        print("-" * 30)
        
        # Method A: Single deterministic run
        print("\nMethod A: Single deterministic run")
        det_answer = solve_problem_deterministic(model, tokenizer, problem_text)
        det_correct = False
        if det_answer is not None:
            # Allow for small floating point differences
            if isinstance(correct_answer, (int, float)) and isinstance(det_answer, (int, float)):
                det_correct = abs(det_answer - correct_answer) < 0.01
            else:
                det_correct = det_answer == correct_answer
                
            if det_correct:
                deterministic_correct += 1
        
        # Method B: Self-consistency with majority vote
        print("\nMethod B: Self-consistency with majority vote")
        sc_answer = solve_problem_with_self_consistency(
            model, tokenizer, problem_text, args.num_samples, args.temperature
        )
        sc_correct = False
        if sc_answer is not None:
            # Allow for small floating point differences
            if isinstance(correct_answer, (int, float)) and isinstance(sc_answer, (int, float)):
                sc_correct = abs(sc_answer - correct_answer) < 0.01
            else:
                sc_correct = sc_answer == correct_answer
                
            if sc_correct:
                self_consistency_correct += 1
        
        print(f"\nCorrect answer: {correct_answer}")
        print(f"Method A correct: {det_correct}")
        print(f"Method B correct: {sc_correct}")
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Method A (Deterministic): {deterministic_correct}/{len(problems)} correct ({deterministic_correct/len(problems)*100:.1f}%)")
    print(f"Method B (Self-consistency): {self_consistency_correct}/{len(problems)} correct ({self_consistency_correct/len(problems)*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
