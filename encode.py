from engram import build_engram
from transformer import get_transformer
from tqdm import tqdm
import pickle
import pandas as pd

# load an example dataset (shakespeare)
# WARNING: This url is only for testing purposes! Please supply your own data
data = pd.read_csv("https://raw.githubusercontent.com/dherman/wc-demo/master/data/shakespeare-plays.csv", error_bad_lines=False)
messages = [ (str(row[1].values[2]), str(row[1].values[3])) for row in data.iterrows() ]

# Load GPT Neo
model, tokenizer = get_transformer()

# encode memories
memories = []
last_speaker = None
full_message = ""
for index in tqdm(range(len(messages))):
    speaker, message = messages[index]
    if last_speaker != speaker and len(full_message) > 0:
        tokens = tokenizer(last_speaker + ": " +full_message.lstrip(), return_tensors="pt").input_ids.cuda()
        memories.append({
            "text": last_speaker + ": " + full_message.lstrip(),
            "engram": build_engram(model.forward, tokens),
            "next": len(memories)+1,
            "previous": len(memories)-1,
            "distance": 0
        })
        full_message = ""
    last_speaker = speaker
    full_message = full_message + " " + message

memories[-1]["next"] = -1

# dump data to disk
with open("shakespeare.pkl", 'wb') as handle:
    pickle.dump(memories, handle, protocol=pickle.HIGHEST_PROTOCOL)
