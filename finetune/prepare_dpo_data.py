import json
from reward_model import score_pair

with open("data/qa_pairs.txt") as f:
    lines = f.read().split("Q: ")[1:]

data = []
for chunk in lines:
    q, rest = chunk.split("A1:", 1)
    a1, a2 = rest.split("A2:")
    q, a1, a2 = q.strip(), a1.strip(), a2.strip()
    
    score1 = score_pair(q, a1)
    score2 = score_pair(q, a2)
    
    if score1 > score2:
        chosen, rejected = a1, a2
    else:
        chosen, rejected = a2, a1

    data.append({"prompt": q, "chosen": chosen, "rejected": rejected})

with open("data/dpo_dataset.json", "w") as f:
    for d in data:
        f.write(json.dumps(d) + "\n")