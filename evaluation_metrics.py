y_true = [1,0,0,1,0]
y_pred = [1,0,0,0,1]

class EvaluationMetrics:
    # Implement Confusion Martix
    print("Confusion Martix \n ========")
    def confusion_martix(y_true, y_pred):
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
    TP,TN,FP,FN = confusion_martix (y_true, y_pred)
    print(f"TP={TP},\n TN={TN},\n FP={FP},\n FN={FN}")

    # Implement Accuracy
    print("\n")
    print("Accuracy \n ========")
    def accuracy(y_true, y_pred) -> float:
        accuracy_value = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return accuracy_value / len(y_true)
    accuracy_result = accuracy(y_true, y_pred)
    print(f"Accuracy: {accuracy_result}")      
    print(f"Accuracy: {accuracy_result:.2f}") 
    print(f"Accuracy: {accuracy_result:.0%}")

    # Implement Precision
    print("\n")
    print("Precision \n ========")
    def precision(y_true, y_pred) -> float:
        TP = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        FP = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        
        if (TP + FP) == 0:        
            return 0.0
        return TP / (TP + FP)
    precesion_value = precision(y_true, y_pred)
    print(f"Precision: {precesion_value}")
    print(f"Precision: {precesion_value:.2f}")
    print(f"Precision: {precesion_value:.0%}")

    # Implement Recall
    print("\n")
    print("Recall \n ========")
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    print(f"Recall = {recall:.2f}")

    # Implement F1 Score
    print("\n")
    print("F1 Score \n ========")
    f1_score = (2 * precesion_value * recall) / (precesion_value + recall) if (precesion_value + recall) > 0 else 0.0
    print(f"F1 Score = {f1_score:.2f}")
    print("\n")


# --- Run ---
ev = EvaluationMetrics()


# Implement Test Cases
print("\n")
print("Test Case 1 — Perfect Classification \n ========")

print("\n")
print("Confusion Matrix \n ========")
y_true = [1, 0, 1, 0]
y_pred = [1, 0, 1, 0]

print("\n")
print("Accuracy \n ========")

print("\n")
print("Precision \n ========")

print("\n")
print("Recall \n ========")

print("\n")
print("F1 Score \n ========")







