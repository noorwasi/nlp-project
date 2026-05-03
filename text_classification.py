dataset = [
    ("free money now", "spam"),
    ("win money now",  "spam"),
    ("call me now",    "normal"),
    ("let's meet now", "normal")
]

def tokenize(text):
    return text.lower().split()

for text, label in dataset:
    print(f"{label} = {tokenize(text)}")

print("\n Compute Word Probabilities \n ===========================")

total_words  = {"spam": 0, "normal": 0}
class_counts = {"spam": 0, "normal": 0}
word_counts  = {"spam": {}, "normal": {}}

# fill the dictionaries
for text, label in dataset:
    tokens = tokenize(text)

    class_counts[label] += 1
   
    total_words[label] += len(tokens)

    for token in tokens:
        if token not in word_counts[label]:
            word_counts[label][token] = 0
        word_counts[label][token] += 1
print(f"class_counts = {class_counts}")
print(f"total_words  = {total_words}")
print(f"word_counts  = {word_counts}")

print("\n Calculate Scoring Function & Fixing the Zero Probablity Issue \n ===========================")
vocab_size = len(set(
    token
    for text, label in dataset
    for token in tokenize(text)
))
def score(text, label):
    tokens = tokenize(text)

    total_docs = sum(class_counts.values())
    p_class = class_counts[label] / total_docs

    p_words = 1.0
    for token in tokens:
        word_count = word_counts[label].get(token, 0)   # 0 if word not found
        p_word     = (word_count + 1) / (total_words[label] + vocab_size)
        p_words    = p_words * p_word
    return p_class * p_words

print(f"score('free money', spam)   = {score('free money', 'spam')}")
print(f"score('free money', normal) = {score('free money', 'normal')}")

print("\n Predict Spam or Normal based on highest Score \n ===========================")
def predict(text):
    spam_score   = score(text, "spam")
    normal_score = score(text, "normal")

    if spam_score > normal_score:
        return "spam"
    else:
        return "normal"

print(f"'win money'- Is this Spam or Normal?   = {predict("win money")}")
print(f"'call me'- Is this Spam or Normal?   = {predict("call me")}")
print(f"'free money'- Is this Spam or Normal?   = {predict("free money")}")
print(f"'let's meet'- Is this Spam or Normal?   = {predict("let's meet")}")
