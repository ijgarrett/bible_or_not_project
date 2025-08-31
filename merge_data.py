import json

with open("niv_dataset.json", "r", encoding = "utf-8") as f:
    niv_data = json.load(f)

with open("non_bible_dataset.json", "r", encoding = "utf-8") as f:
    non_bible_data = json.load(f)

combined = niv_data + non_bible_data # merge the two lists of dicts

with open("combined_dataset.json", "w", encoding = "utf-8") as f:
    json.dump(combined, f, ensure_ascii = False, indent = 2)

print(f"Combined dataset size: {len(combined)}")