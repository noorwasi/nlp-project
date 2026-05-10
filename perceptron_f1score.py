# IMDB Movie Reviews Classifier
import csv
import random

# Load IMDb Data
# =========================================
dataset = []
with open("imdb_movie_reviews.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text  = row['review'].replace('<br />', ' ')
        label = 1 if row['sentiment'] == "positive" else -1
        dataset.append((text, label))

# shuffle and pick 500 reviews
random.seed(67)
random.shuffle(dataset)
dataset = dataset[:500]

# split into train (80) and test (20)
train_data = dataset[:400]
test_data  = dataset[400:]

print("IMDb Perceptron Classifier")
print("========")
print(f"Training samples = {len(train_data)}")
print(f"Test samples     = {len(test_data)}")


# Tokenize
# =========================================
def tokenize(text):
    return text.lower().split()


# Build Vocab
# =========================================
def build_vocab(dataset):
    vocab = set()
    for text, label in dataset:
        tokens = tokenize(text)
        for token in tokens:
            vocab.add(token)
    return sorted(vocab)

vocab = build_vocab(train_data)
print(f"\n Vocab \n ===========================")
print(f"vocab size = {len(vocab)}")


# Vectorize
# =========================================
def vectorize(text):
    tokens = tokenize(text)
    vector = []
    for word in vocab:
        if word in tokens:
            vector.append(1)
        else:
            vector.append(0)
    return vector


# Initialize Weights and Bias
# =========================================
w = [0] * len(vocab)
b = 0

print(f"\n Initial Weights and Bias \n ===========================")
print(f"weights = {w[:10]} ...")
print(f"bias    = {b}")


# Predict
# =========================================

def predict(x):
    score = 0
    for i in range(len(w)):
        score = score + w[i] * x[i]
    score = score + b
    if score > 0:
        return 1
    else:
        return -1


# Update Weights
# =========================================
def update(x, y):
    global b
    predicted = predict(x)
    if predicted != y:
        for i in range(len(w)):
            w[i] = w[i] + y * x[i]
        b = b + y


# Training Loop
# =========================================
print(f"\n Training \n ===========================")
epochs = 10
for _ in range(epochs):
    for text, label in train_data:
        x = vectorize(text)
        update(x, label)

print(f"bias after training    = {b}")
print(f"weights after training = {w[:10]} ...")


# Predict on Test Data
# =========================================

print(f"\n Predictions on Test Data \n ===========================")

y_true = []
y_pred = []

for text, true_label in test_data:
    x               = vectorize(text)
    predicted_label = predict(x)
    y_true.append(true_label)
    y_pred.append(predicted_label)
    true_str      = "positive" if true_label      ==  1 else "negative"
    predicted_str = "positive" if predicted_label ==  1 else "negative"
    print(f"true = {true_str:8} predicted = {predicted_str:8} review = {text[:40]}...")


# Evaluation
# =========================================
class EvaluationMetrics:

    def confusion_matrix(self, y_true, y_pred):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for t, p in zip(y_true, y_pred):
            if t == 1  and p == 1:
                TP += 1
            if t == -1 and p == -1:
                TN += 1
            if t == -1 and p == 1:
                FP += 1
            if t == 1  and p == -1:
                FN += 1
        return TP, TN, FP, FN

    def accuracy(self, y_true, y_pred):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    def precision(self, y_true, y_pred):
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == 1)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t == -1 and p == 1)
        if (TP + FP) == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self, y_true, y_pred):
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == 1)
        FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1  and p == -1)
        if (TP + FN) == 0:
            return 0.0
        return TP / (TP + FN)

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        if (p + r) == 0:
            return 0.0
        return (2 * p * r) / (p + r)


# --- Run Evaluation ---
ev = EvaluationMetrics()

TP, TN, FP, FN = ev.confusion_matrix(y_true, y_pred)

print(f"\n Confusion Matrix \n ===========================")
print(f"TP = {TP}")
print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}")

print(f"\n Accuracy \n ===========================")
accuracy_result = ev.accuracy(y_true, y_pred)
print(f"Accuracy = {accuracy_result:.2f}")

print(f"\n Precision \n ===========================")
precision_value = ev.precision(y_true, y_pred)
print(f"Precision = {precision_value:.2f}")

print(f"\n Recall \n ===========================")
recall_value = ev.recall(y_true, y_pred)
print(f"Recall = {recall_value:.2f}")

print(f"\n F1 Score \n ===========================")
f1_value = ev.f1_score(y_true, y_pred)
print(f"F1 Score = {f1_value:.2f}")