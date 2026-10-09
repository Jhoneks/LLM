import argparse
import pickle

import torch

# pickle looks these classes up by name, so they must exist in this script
from training import (Head, MultiHeadAttention, FeedForward, Block, GPTLanguageModel,
                      encode, decode, device, model_path)

parser = argparse.ArgumentParser(description='Chat with the trained GPT')
parser.add_argument('-max_new_tokens', type=int, default=150, help='characters to generate per reply')
args, _ = parser.parse_known_args()

print('loading model parameters...')
with open(model_path, 'rb') as f:
    model = pickle.load(f)
model = model.to(device)
model.eval()
print('loaded successfully! Type a prompt, or "quit" to exit.')

while True:
    prompt = input('\nPrompt:\n').lstrip('﻿')  # piped input on Windows can start with a BOM
    if prompt.strip().lower() in ('quit', 'exit'):
        break
    tokens = encode(prompt) or encode('\n')
    context = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=args.max_new_tokens)[0].tolist()
    print(f'Completion:\n{decode(generated)}')
