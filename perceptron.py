# Dataset with 1 as Spam and -1 as Normal
dataset = [
 ("free money now", 1),
 ("win money now", 1),
 ("call me now", -1),
 ("let's meet now", -1)
]
# Tokenize
def tokenize(text):
    return text.lower().split()

# Build Vocab 
def build_vocab(dataset):
    vocab = set()
    for text, label in dataset:
        tokens = tokenize(text)
        for token in tokens:
            vocab.add(token)
    return sorted(vocab)

# Show vocab and vocab size
vocab = build_vocab(dataset)
print(f"vocab size = {len(vocab)}")
print(f"vocab      = {vocab}")

# Vectorize The dataset
def vectorize(text):
    tokens = tokenize(text)
    vector = []
    for word in vocab:
        if word in tokens:
            vector.append(1)
        else:
            vector.append(0)
    return vector

print("\n Vectorized Dataset \n ===========================")
print(vectorize("free money"))
print(vectorize("call me now"))


print("\n Prediction & Weights Correction \n ===========================")
# Initialize weights and bias
w = [0] * len(vocab)
b = 0

print(f"\n Initial Weights and Bias \n ===========================")
print(f"weights = {w}")
print(f"bias    = {b}")

# Predict
def predict(x):
    score = 0
    for i in range(len(w)):
        score = score + w[i] * x[i]
    score = score + b
    if score > 0:
        return 1
    else:
        return -1

# Update
def update(x, y):
    global b
    predicted = predict(x)
    if predicted != y:
        for i in range(len(w)):
            w[i] = w[i] + y * x[i]
        b = b + y

# Training Loop
print(f"\n Training \n ===========================")
epochs = 10
for _ in range(epochs):
    for text, label in dataset:
        x = vectorize(text)
        update(x, label)

print(f"weights after training = {w}")
print(f"bias after training    = {b}")

# Test the model
print(f"\n Test Predictions \n ===========================")
tests = ["free money", "call me", "meet now"]
for t in tests:
    x = vectorize(t)
    print(f"{t} = {predict(x)}")