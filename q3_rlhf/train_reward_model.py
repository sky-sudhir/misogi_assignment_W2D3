import pandas as pd
from datasets import Dataset
from transformers import GPT2Tokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
import matplotlib.pyplot as plt
import torch
from transformers import pipeline, GPT2LMHeadModel
import itertools

# 1. Load ranked data and convert to preference pairs
df = pd.read_csv("answers_ranked.csv")
pairs = []
for prompt, group in df.groupby("prompt"):
    answers = group.sort_values("rank")
    for (_, row_chosen), (_, row_rejected) in itertools.permutations(answers.iterrows(), 2):
        if row_chosen["rank"] < row_rejected["rank"]:
            pairs.append({
                "prompt": prompt,
                "chosen": row_chosen["answer"],
                "rejected": row_rejected["answer"]
            })
pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv("answers_pairs.csv", index=False)
print(pairs_df.head())

# 2. Prepare dataset for reward model
dataset = Dataset.from_pandas(pairs_df)

# 3. Train reward model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Fix 1: Properly set padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

reward_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Fix 2: Resize token embeddings to handle the new padding token properly
reward_model.resize_token_embeddings(len(tokenizer))

# Fix 3: Configure the model to handle padding
reward_model.config.pad_token_id = tokenizer.pad_token_id

config = RewardConfig(
    output_dir="./reward_model",
    per_device_train_batch_size=1,  # Fix 4: Reduce batch size to 1 to avoid issues
    num_train_epochs=1,
    max_steps=100,
    logging_steps=10,
    save_steps=50,
    bf16=False,
    fp16=False,
    remove_unused_columns=False,  # Fix 5: Keep all columns for RewardTrainer
    dataloader_pin_memory=False,  # Fix 6: Disable pin_memory to avoid warning
)

trainer = RewardTrainer(
    model=reward_model,
    args=config,
    train_dataset=dataset,
    eval_dataset=None,
    processing_class=tokenizer
)

# Fix 7: Handle potential CUDA issues
if torch.cuda.is_available():
    reward_model = reward_model.cuda()

trainer.train()

# 4. Evaluate and plot reward scores
generator = pipeline("text-generation", model=GPT2LMHeadModel.from_pretrained(model_name), tokenizer=tokenizer)
new_prompt = "Describe a beautiful sunset."
outputs = generator(new_prompt, max_length=60, num_return_sequences=4, do_sample=True)
new_answers = [out['generated_text'] for out in outputs]

# Fix 8: Properly format inputs for reward model evaluation
reward_model.eval()
inputs = tokenizer([f"{new_prompt} {ans}" for ans in new_answers], 
                  return_tensors="pt", 
                  padding=True, 
                  truncation=True, 
                  max_length=512)

with torch.no_grad():
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Fix 9: Get scalar reward scores instead of softmax
    scores = reward_model(**inputs).logits.squeeze(-1)  # Remove softmax, get raw scores
    if torch.cuda.is_available():
        scores = scores.cpu()
    reward_scores = scores.numpy()

plt.figure(figsize=(10, 6))
plt.bar(range(1, len(reward_scores) + 1), reward_scores)
plt.xlabel("Answer")
plt.ylabel("Reward Score")
plt.title("Reward Scores for New Answers")
plt.xticks(range(1, len(reward_scores) + 1))
plt.tight_layout()
plt.show()

print("\nGenerated answers with their reward scores:")
print("=" * 50)
for i, (ans, score) in enumerate(zip(new_answers, reward_scores), 1):
    print(f"Answer {i} (Score: {score:.4f}):")
    print(f"{ans}")
    print("-" * 30)