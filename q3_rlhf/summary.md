# Reward Model Training Project

## Overview

This project implements a reward modeling system using the TRL (Transformer Reinforcement Learning) library. The reward model is trained to learn preferences between different text responses, which can later be used for reinforcement learning tasks like RLHF (Reinforcement Learning from Human Feedback).

## Components

### 1. Data Processing (`train_reward_model.py`)

- Reads ranked answers from `answers_ranked.csv`
- Converts rankings into preference pairs (chosen vs rejected)
- Creates `answers_pairs.csv` with columns:
  - `prompt`: The original question/prompt
  - `chosen`: The preferred answer
  - `rejected`: The less preferred answer

### 2. Model Architecture

- Base model: GPT-2
- Modified for sequence classification (reward prediction)
- Single output neuron for preference scoring
- Properly configured padding tokens and embeddings

### 3. Training Configuration

- Uses TRL's `RewardTrainer`
- Training parameters:
  - Batch size: 1 (to avoid padding issues)
  - Number of epochs: 1
  - Maximum steps: 100
  - Logging frequency: Every 10 steps
  - Model saving: Every 50 steps
  - Mixed precision: Disabled (no bf16/fp16)

### 4. Evaluation

- Generates new responses using GPT-2
- Scores responses using the trained reward model
- Visualizes scores using matplotlib
- Prints detailed comparison of generated answers with their scores

## Usage

1. Prepare your data:

   ```
   answers_ranked.csv should contain:
   - prompt: The question or context
   - answer: The response text
   - rank: Numerical ranking (lower is better)
   ```

2. Run the training:

   ```bash
   python train_reward_model.py
   ```

3. Results:
   - Trained model saved in `./reward_model/`
   - Preference pairs saved in `answers_pairs.csv`
   - Evaluation plots showing reward scores for generated responses

## Dependencies

- transformers
- trl
- torch
- pandas
- matplotlib
- datasets

## Notes

- The model uses GPT-2's EOS token as padding token
- Token embeddings are resized to handle padding
- CUDA support is implemented when available
- Evaluation includes both numerical scores and visualizations

## Future Improvements

1. Implement cross-validation
2. Add support for different base models
3. Implement hyperparameter tuning
4. Add more evaluation metrics
5. Support for larger batch sizes with proper padding
