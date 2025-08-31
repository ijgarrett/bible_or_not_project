import json
import random
import wikipedia
import re

# Load the nested JSON file
with open("niv_bible.json", "r", encoding="utf-8") as f:
    nested_data = json.load(f)

# Prepare flat dataset
flat_passages = []

for book, chapters in nested_data.items():
    for chapter, verses in chapters.items():
        for verse_num, verse_text in verses.items():
            flat_passages.append({
                "book": book,
                "text": verse_text.strip(),
                "label": 1
            })

# Save to a new JSON file
with open("niv_dataset.json", "w", encoding="utf-8") as f:
    json.dump(flat_passages, f, indent=2, ensure_ascii=False)

print(f"Dataset created! Total passages: {len(flat_passages)}") # extra verse in 3 John, 15 instead 14
