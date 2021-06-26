import os
import pickle
from engram import build_engram, sort_engrams
from transformer import get_transformer, get_generator

# Load GPT Neo
model, tokenizer = get_transformer()
generate = get_generator(model, tokenizer)

# Change as needed, this works with the shakespere example dataset in encode.py
memory_file = "shakespeare.pkl"
speaker_name = "JULIET"
GPT_name = "ROMEO"

print("Welcome to GPT Chat! Load up your dataset")
print(f"You are chatting as {speaker_name}, and GPT is chatting as {GPT_name}.")

context = []
memories = []

if os.path.exists(memory_file):
    with open(memory_file, 'rb') as handle:
        memories = pickle.load(handle)

def add_engram(text, add_context=True):
    context.append(text)
    memoryCount = len(memories)
    memories[-1]["next"] = memoryCount
    engram = {
        "text": text,
        "engram": build_engram(model.forward, tokenizer(text, return_tensors="pt").input_ids.cuda()),
        "next": -1,
        "previous": memoryCount-1,
        "distance": 0
    }
    memories.append(engram)
    return engram

def build_context(now, short_term=10):
    # sort engrams
    m = sort_engrams(now, memories[:-short_term], top_k=600)
    m = sort_engrams(now, m, top_k=150, do_distance=False, depth=2)
    m = sort_engrams(now, m, top_k=42, do_distance=False, depth=3)
    m.reverse()

    text = ""

    for memory in m:
        if not memory["text"].startswith(speaker_name):
            text = text + memories[memory["previous"]]["text"] + "\n"
        text = text + memory["text"] + "\n"
        if memory["text"].startswith(speaker_name):
            text = text + memories[memory["next"]]["text"] + "\n"

    for recent in context[-short_term:]: # 10 most recent messages
        text = text + recent + "\n"
    
    return text

while True:
    # let user input a message
    message = input(speaker_name + ": ")

    engram = add_engram(speaker_name + ": " + message)

    text = build_context(engram) + GPT_name + ":"

    reply = generate(text).split("\n")[0]
    print(GPT_name + ":" + reply)
    
    add_engram(GPT_name + ":" + reply)
