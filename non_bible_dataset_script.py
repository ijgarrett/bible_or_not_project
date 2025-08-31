import json
import regex as re
import random
import requests
from datetime import datetime

# 66 public domain books from Project Gutenberg (a mix of classics)
BOOK_URLS = [
    "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "https://www.gutenberg.org/files/84/84-0.txt",      # Frankenstein
    "https://www.gutenberg.org/files/11/11-0.txt",      # Alice in Wonderland
    "https://www.gutenberg.org/files/1661/1661-0.txt",  # Sherlock Holmes
    "https://www.gutenberg.org/files/98/98-0.txt",      # Tale of Two Cities
    "https://www.gutenberg.org/files/120/120-0.txt",    # Treasure Island
    "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
    "https://www.gutenberg.org/files/5200/5200-0.txt",  # Metamorphosis
    "https://www.gutenberg.org/files/74/74-0.txt",      # Tom Sawyer
    "https://www.gutenberg.org/files/345/345-0.txt",    # Dracula
    "https://www.gutenberg.org/files/45/45-0.txt",      # Anne of Green Gables
    "https://www.gutenberg.org/files/43/43-0.txt",      # Sherlock Holmes #2
    "https://www.gutenberg.org/files/514/514-0.txt",    # Little Women
    "https://www.gutenberg.org/files/1080/1080-0.txt",  # A Modest Proposal
    "https://www.gutenberg.org/files/100/100-0.txt",    # Divine Comedy
    "https://www.gutenberg.org/files/25344/25344-0.txt",# A Doll's House
    "https://www.gutenberg.org/files/3289/3289-0.txt",  # Meditations (Marcus Aurelius)
    "https://www.gutenberg.org/files/105/105-0.txt",    # Persuasion
    "https://www.gutenberg.org/files/8800/8800-0.txt",  # Flatland
    "https://www.gutenberg.org/files/28054/28054-0.txt",# Beowulf
    "https://www.gutenberg.org/files/4363/4363-0.txt",  # Beyond Good and Evil
    "https://www.gutenberg.org/files/766/766-0.txt",    # Gulliver's Travels
    "https://www.gutenberg.org/files/2148/2148-0.txt",  # The Prince
    "https://www.gutenberg.org/files/174/174-0.txt",    # The Picture of Dorian Gray
    "https://www.gutenberg.org/files/27761/27761-0.txt",# The Communist Manifesto
    "https://www.gutenberg.org/files/996/996-0.txt",    # Don Quixote (Part 1)
    "https://www.gutenberg.org/files/59467/59467-0.txt",# Don Quixote (Part 2)
    "https://www.gutenberg.org/files/42108/42108-0.txt",# Utopia
    "https://www.gutenberg.org/files/19942/19942-0.txt",# Confessions (St. Augustine)
    "https://www.gutenberg.org/files/408/408-0.txt",    # The Federalist Papers
    "https://www.gutenberg.org/files/2000/2000-0.txt",  # Grimm's Fairy Tales
    "https://www.gutenberg.org/files/15/15-0.txt",      # Wizard of Oz
    "https://www.gutenberg.org/files/55/55-0.txt",      # Wizard of Oz #2
    "https://www.gutenberg.org/files/289/289-0.txt",    # Essays by Emerson
    "https://www.gutenberg.org/files/27827/27827-0.txt",# Hamlet
    "https://www.gutenberg.org/files/1524/1524-0.txt",  # Othello
    "https://www.gutenberg.org/files/1777/1777-0.txt",  # Antony and Cleopatra
    "https://www.gutenberg.org/files/10007/10007-0.txt",# Macbeth
    "https://www.gutenberg.org/files/2264/2264-0.txt",  # Much Ado About Nothing
    "https://www.gutenberg.org/files/11030/11030-0.txt",# Romeo and Juliet
    "https://www.gutenberg.org/files/1339/1339-0.txt",  # King Lear
    "https://www.gutenberg.org/files/203/203-0.txt",    # Cyrano de Bergerac
    "https://www.gutenberg.org/files/244/244-0.txt",    # The Jungle
    "https://www.gutenberg.org/files/2147/2147-0.txt",  # The Tempest
    "https://www.gutenberg.org/files/1260/1260-0.txt",  # Jane Eyre
    "https://www.gutenberg.org/files/768/768-0.txt",    # Wuthering Heights
    "https://www.gutenberg.org/files/1952/1952-0.txt",  # The Yellow Wallpaper
    "https://www.gutenberg.org/files/16328/16328-0.txt",# Walden
    "https://www.gutenberg.org/files/1200/1200-0.txt",  # Paradise Lost
    "https://www.gutenberg.org/files/7370/7370-0.txt",  # Leviathan (Hobbes)
    "https://www.gutenberg.org/files/20203/20203-0.txt",# On Liberty (Mill)
    "https://www.gutenberg.org/files/20899/20899-0.txt",# Utilitarianism
    "https://www.gutenberg.org/files/24280/24280-0.txt",# Democracy in America
    "https://www.gutenberg.org/files/132/132-0.txt",    # The Art of War
    "https://www.gutenberg.org/files/10552/10552-0.txt",# Paradise Regained
    "https://www.gutenberg.org/files/11231/11231-0.txt",# Common Sense (Paine)
    "https://www.gutenberg.org/files/10681/10681-0.txt",# The Republic (Plato)
    "https://www.gutenberg.org/files/1497/1497-0.txt",  # The Symposium (Plato)
    "https://www.gutenberg.org/files/2814/2814-0.txt",  # The Iliad
    "https://www.gutenberg.org/files/1727/1727-0.txt",  # The Odyssey
    "https://www.gutenberg.org/files/6130/6130-0.txt",  # The Aeneid
    "https://www.gutenberg.org/files/4517/4517-0.txt",  # Faust
    "https://www.gutenberg.org/files/16389/16389-0.txt",# Candide
    "https://www.gutenberg.org/files/37833/37833-0.txt",# Federal Constitution
    "https://www.gutenberg.org/files/7371/7371-0.txt",  # The Social Contract
    "https://www.gutenberg.org/files/4361/4361-0.txt",  # Ethics (Spinoza)
    "https://www.gutenberg.org/files/17192/17192-0.txt" # Thus Spake Zarathustra
]

def smart_sentence_split(text):
    # holds common abbreviations
    abbreviations = {
        "Mr", "Mrs", "Ms", "Miss", "Dr", "Prof", "Sr", "Jr", "St", "vs", "etc", 
        "Inc", "Ltd", "Corp", "Co", "Ave", "Blvd", "Rd", "Str", "Ph", "B.A",
        "M.A", "Ph.D", "B.S", "M.S", "D.D.S", "M.D", "J.D", "LL.B", "LL.M",
        "i.e", "e.g", "cf", "al", "ibid", "op", "loc", "ca", "c", "approx",
        "vol", "no", "p", "pp", "ch", "sec", "fig", "figs", "refs", "ref",
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    }
    
    protected_text = text
    for abbr in abbreviations:
        # re.sub(pattern, replacement, string, flags=...)
        # replaces all instances of pattern with replacement. flags = re.IGNORECASE ignores upper vs lower case
        # r = raw string literal(backslashes don't need escaping)
        # \b is a regex word boundary: ensures abbreviation appears as a whole word
        # re.escape(abbr) escapes special regex chars indside abbr if any
        # \. is a literal dot (.). In regex, . normally means any character
        # replace "Mr." with "MrDOTPLACEHOLDER"
        protected_text = re.sub(rf"\b{re.escape(abbr)}\.", f"{abbr}DOTPLACEHOLDER", protected_text, flags=re.IGNORECASE)
    
    # (?<=[.!?]): positive lookbehind. the position must be preceded by a ., !, or ?
    # (?:\s*[\"'\)]*\s*): 
        # (?:...) non-capturing group
        # \s* = 0 or more whitespace characters
        # [\"'\)]* = 0 or more of ", ', or )
        # \s* = 0 or more whitespace again
        # + one or more of entire group
    # (?=[\"'\([]*[A-Z]) = postive lookahead, the position must be followed by:
    # any number of quotes or brackets, but eventually you must hit a capital letter
    sentence_pattern = r"(?<=[.!?])(?:\s*[\"'\)]*\s*)+(?=[\"'\([]*[A-Z])"
    # split the text
    sentences = re.split(sentence_pattern, protected_text)
    
    # replace the placeholder with a period
    sentences = [s.replace("DOTPLACEHOLDER", ".") for s in sentences]
    
    return sentences

def clean_and_filter_sentences(sentences):
    cleaned = []
    for s in sentences:
        # \s+ means one or more whitespace characters, substitute with a single space
        # .strip() removes whitespace at beginning and end
        s = re.sub(r"\s+", " ", s).strip()
        
        if len(s.split()) <= 5:
            # if less than 5 words, skip
            continue
        if len(s.split()) > 100:
            # if more than 100 words, skip
            continue
        if not re.match(r"^[A-Z\"']", s):
            # re.match(pattern, string) checks if pattern matches at the beginning of the string
            # ^ means start with
            # setence must start with capital letter or quotes
            continue
        if not re.search(r"[.!?][\"']*$", s):
            # re.search looks anywhere in s for a match
            # [.!?\] = match one char that is either of those
            # [\"']* = match zero or more quotes
            # $ = anchor for end of string
            continue
        if re.match(r"^(and|but|or|so|yet|for|nor|because|since|although|though|if|when|where|while|after|before|until)\b", s, re.IGNORECASE):
            # if sentence starts with any of these (\b means boundary after the word), skip
            continue
        if re.match(r"^(CHAPTER|Chapter|chapter)\s+[IVX\d]+", s):
            # skip sentences that start with "chapter", any number of whitespace, and roman numerals or digits (\d)
            continue
        if len(s) < 20:
            # if less than 20 characters, skip
            continue
            
        cleaned.append(s)
    
    return cleaned

negative_passages = []
limit_per_book = 600
successful_books = 0

for url in BOOK_URLS:
    print(f"Downloading {url}...")
    try:
        # sends HTTP GET request to the URL
        # wait up to 20 seconds for a response before raising an error
        response = requests.get(url, timeout=20)
        # if server returns error, it raises an exception
        response.raise_for_status()
        text = response.text
    except Exception as e:
        print(f"Failed {url}: {e}")
        continue
    
    start = text.find("*** START")
    end = text.find("*** END")
    if start != -1 and end != -1:
        text = text[start:end]
    elif text.find("***START") != -1:
        # alternate format without the space
        start = text.find("***START")
        end = text.find("***END")
        if start != -1 and end != -1:
            text = text[start:end]
    
    # \n makes sure the match starts right after a new line
    # \s* allows any number of whitespace chars
    # Chapter + roman numerals or digits
    # .*? matches anything else in that line, like a subtitle
        # .* means any number of any chars, ? makes it lazy: grabs as little as possible before the next \n
    # followed with a newline
    # replace with just a newline
    text = re.sub(r"\n\s*CHAPTER [IVX\d]+.*?\n", "\n", text, flags = re.IGNORECASE)
    
    sentences = smart_sentence_split(text)
    cleaned = clean_and_filter_sentences(sentences)
    
    print(f"  Found {len(cleaned)} valid sentences")
    
    if len(cleaned) < 50:
        print("  Skipping - too few valid sentences")
        continue
    
    if len(cleaned) >= limit_per_book:
        sampled = random.sample(cleaned, limit_per_book)
    else:
        sampled = cleaned
    
    book_name = url.split("/")[-1]
    for s in sampled:
        negative_passages.append({
            "book": book_name,
            "text": s,
            "label": 0,
        })
    
    successful_books += 1
    print(f"  Added {len(sampled)} sentences from {book_name}")

print("\nSummary:")
print(f"Successfully processed {successful_books}/{len(BOOK_URLS)} books")
print(f"Total collected passages: {len(negative_passages)}")

fname = f"non_bible_dataset_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
with open(fname, "w", encoding="utf-8") as f:
    json.dump(negative_passages, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {fname}")
