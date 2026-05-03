import csv
import random

# Load IMDb Data
# =========================================

dataset = []
with open("imdb_movie_reviews.csv", newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        text  = row['review'].replace('<br />', ' ')
        label = row['sentiment']
        dataset.append((text, label))

# shuffle and pick 500 reviews
random.seed(67)
random.shuffle(dataset)
dataset = dataset[:500]

# split into train (80) and test (20) rule
train_data = dataset[:400]
test_data  = dataset[400:]



print("IMDb Reviews Classifier")
print("========")
print(f"Training samples = {len(train_data)}")
print(f"Test samples     = {len(test_data)}")

# Tokenizer
# =========================================

def tokenize(text):
    return text.lower().split()



# Train — fill word counts
# =========================================

class_counts = {"positive": 0, "negative": 0}
total_words  = {"positive": 0, "negative": 0}
word_counts  = {"positive": {}, "negative": {}}

for text, label in train_data:
    tokens = tokenize(text)
    class_counts[label] += 1
    total_words[label]  += len(tokens)
    for token in tokens:
        if token not in word_counts[label]:
            word_counts[label][token] = 0
        word_counts[label][token] += 1

print("\n Word Counts")
print("========")
print(f"class_counts = {class_counts}")
print(f"total_words  = {total_words}")


# Vocab Size
# =========================================

vocab_size = len(set(
    token
    for text, label in train_data
    for token in tokenize(text)
))

print(f"vocab_size   = {vocab_size}")

# Score Function
# =========================================

def score(text, label):
    tokens     = tokenize(text)
    total_docs = sum(class_counts.values())
    p_class    = class_counts[label] / total_docs
    p_words    = 1.0
    for token in tokens:
        word_count = word_counts[label].get(token, 0)
        p_word     = (word_count + 1) / (total_words[label] + vocab_size)
        p_words    = p_words * p_word
    return p_class * p_words


# Predict Function
# =========================================

def predict(text):
    pos_score = score(text, "positive")
    neg_score = score(text, "negative")
    if pos_score > neg_score:
        return "positive"
    else:
        return "negative"

# Predict on Test Data
# =========================================

print("\n Predictions on Test Data")
print("========")

y_true = []
y_pred = []

for text, true_label in test_data:
    predicted_label = predict(text)
    y_true.append(1 if true_label == "positive" else 0)
    y_pred.append(1 if predicted_label == "positive" else 0)
    print(f"true = {true_label:8} predicted = {predicted_label:8} review = {text[:40]}...")


# Evaluation of Metrics
# =========================================

class EvaluationMetrics:

    def confusion_matrix(self, y_true, y_pred):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for t, p in zip(y_true, y_pred):
            if t == 1 and p == 1:
                TP += 1
            if t == 0 and p == 0:
                TN += 1
            if t == 0 and p == 1:
                FP += 1
            if t == 1 and p == 0:
                FN += 1
        return TP, TN, FP, FN

    def accuracy(self, y_true, y_pred):
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return correct / len(y_true)

    def precision(self, y_true, y_pred):
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        if (TP + FP) == 0:
            return 0.0
        return TP / (TP + FP)

    def recall(self, y_true, y_pred):
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        FN = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        if (TP + FN) == 0:
            return 0.0
        return TP / (TP + FN)

    def f1_score(self, y_true, y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        if (p + r) == 0:
            return 0.0
        return (2 * p * r) / (p + r)


# --- Run Evaluation class ---
ev = EvaluationMetrics()

TP, TN, FP, FN = ev.confusion_matrix(y_true, y_pred)

print("\n Confusion Matrix")
print("========")
print(f"TP = {TP}")
print(f"TN = {TN}")
print(f"FP = {FP}")
print(f"FN = {FN}")

print("\n Accuracy")
print("========")
accuracy_result = ev.accuracy(y_true, y_pred)
print(f"Accuracy = {accuracy_result:.2f}")

print("\n Precision")
print("========")
precision_value = ev.precision(y_true, y_pred)
print(f"Precision = {precision_value:.2f}")

print("\n Recall")
print("========")
recall_value = ev.recall(y_true, y_pred)
print(f"Recall = {recall_value:.2f}")

print("\n F1 Score")
print("========")
f1_value = ev.f1_score(y_true, y_pred)
print(f"F1 Score = {f1_value:.2f}")


print("\n Checking the Data Sample")
print("========")
pos_in_test = sum(1 for text, label in test_data if label == "positive")
neg_in_test = sum(1 for text, label in test_data if label == "negative")
print(f"positive in test = {pos_in_test}")
print(f"negative in test = {neg_in_test}")