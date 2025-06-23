import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# 1. Pick 5 prompts
prompts = [
    "Tell me a joke about computers.",
    "Summarize the importance of exercise.",
    "Write a short story about a lost dog.",
    "Explain how photosynthesis works.",
    "Describe your favorite season and why."
]

# 2. Generate 4 candidate answers per prompt
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

all_data = []
for prompt in prompts:
    outputs = generator(prompt, max_length=60, num_return_sequences=4, do_sample=True)
    for out in outputs:
        all_data.append({"prompt": prompt, "answer": out['generated_text'], "rank": None})

df = pd.DataFrame(all_data)
df.to_csv("answers.csv", index=False)
print("answers.csv saved. Now you can rank the answers interactively.")

# 3. Interactive ranking
for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    answers = df[df['prompt'] == prompt]['answer'].tolist()
    for idx, ans in enumerate(answers, 1):
        print(f"\nAnswer {idx}:")
        print(ans)
    print("\nPlease assign a unique rank (1=best, 4=worst) to each answer.")
    ranks = []
    while True:
        try:
            ranks = [int(input(f"Rank for Answer {i}: ")) for i in range(1, 5)]
            if sorted(ranks) == [1, 2, 3, 4]:
                break
            else:
                print("Ranks must be unique and between 1 and 4. Please try again.")
        except ValueError:
            print("Invalid input. Please enter integers 1-4.")
    # Update ranks in DataFrame
    df.loc[df['prompt'] == prompt, 'rank'] = ranks

# Save ranked answers
df.to_csv("answers_ranked.csv", index=False)
print("\nRanking complete. Ranked answers saved to answers_ranked.csv.") 