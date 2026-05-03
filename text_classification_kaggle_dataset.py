# IMDB Movie Reviews Classifier
import csv
import random

# STEP 1 - Load a small sample of the data
# =========================================

def load_imdb(filepath, sample_size=50, seed=42):
    data = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # clean the review text — remove basic HTML tags
            text  = row['review'].replace('<br />', ' ')
            label = row['sentiment']          # "positive" or "negative"
            data.append((text, label))

    # shuffle so we get a mix of positive and negative
    random.seed(seed)
    random.shuffle(data)
    return data[:sample_size]

# STEP 2 - Tokenizer
# =========================================

def tokenize(text):
    return text.lower().split()

# STEP 3 - Train Naive Bayes
# =========================================

def train(dataset):
    class_counts = {"positive": 0, "negative": 0}
    total_words  = {"positive": 0, "negative": 0}
    word_counts  = {"positive": {}, "negative": {}}

    for text, label in dataset:
        tokens = tokenize(text)
        class_counts[label] += 1
        total_words[label]  += len(tokens)
        for token in tokens:
            if token not in word_counts[label]:
                word_counts[label][token] = 0
            word_counts[label][token] += 1

    return class_counts, total_words, word_counts


# STEP 4 - Score and Predict
# =========================================

def score(text, label, class_counts, total_words, word_counts, vocab_size):
    tokens = tokenize(text)

    # P(c)
    total_docs = sum(class_counts.values())
    p_class    = class_counts[label] / total_docs

    # P(w|c) with laplace smoothing
    p_words = 1.0
    for token in tokens:
        word_count = word_counts[label].get(token, 0)
        p_word     = (word_count + 1) / (total_words[label] + vocab_size)
        p_words    = p_words * p_word

    return p_class * p_words


def predict(text, class_counts, total_words, word_counts, vocab_size):
    pos_score = score(text, "positive", class_counts, total_words, word_counts, vocab_size)
    neg_score = score(text, "negative", class_counts, total_words, word_counts, vocab_size)
    if pos_score > neg_score:
        return "positive"
    else:
        return "negative"

# STEP 5 - Evaluator (your code from before)
# =========================================

class Evaluator:

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

# STEP 6 - Run Everything
# =========================================

# --- load data ---
dataset = load_imdb("imdb_movie_reviews.csv", sample_size=500)

# --- split into train (80) and test (20) rule ---
train_data = dataset[:400]
test_data  = dataset[400:]

# --- train ---
class_counts, total_words, word_counts = train(train_data)

# --- build vocab from training data only ---
vocab_size = len(set(
    token
    for text, label in train_data
    for token in tokenize(text)
))

print(f"Training samples : {len(train_data)}")
print(f"Test samples     : {len(test_data)}")
print(f"Vocab size       : {vocab_size}")
print()

# --- predict on test data ---
y_true = []
y_pred = []

for text, true_label in test_data:
    predicted_label = predict(text, class_counts, total_words, word_counts, vocab_size)

    # convert labels to 1/0 for evaluator
    # positive = 1, negative = 0
    y_true.append(1 if true_label    == "positive" else 0)
    y_pred.append(1 if predicted_label == "positive" else 0)

# --- evaluate ---
ev = Evaluator()

TP, TN, FP, FN = ev.confusion_matrix(y_true, y_pred)
print("\n Results on IMDb sample")
print("======================")
print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")
print(f"Accuracy  : {ev.accuracy(y_true, y_pred):.2f}")
print(f"Precision : {ev.precision(y_true, y_pred):.2f}")
print(f"Recall    : {ev.recall(y_true, y_pred):.2f}")
print(f"F1 Score  : {ev.f1_score(y_true, y_pred):.2f}")

# --- show predictions on test reviews ---
print("\n Sample Predictions")
print("======================")
for text, true_label in test_data:
    predicted = predict(text, class_counts, total_words, word_counts, vocab_size)
    match     = "Yes" if predicted == true_label else "No"
    print(f"{match} , true = {true_label:8} , predicted = {predicted:8} : review = {text[:50]}...")

print("\n Check the Data Sample for total positives and negatives")
print("======================")
pos_in_test = sum(1 for text, label in test_data if label == "positive")
neg_in_test = sum(1 for text, label in test_data if label == "negative")
print(f"count positive in test = {pos_in_test}")
print(f"count negative in test = {neg_in_test}")