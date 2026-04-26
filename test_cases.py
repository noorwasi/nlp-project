# Implement Tests

class EvaluationTests:

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
        accuracy_value = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        return accuracy_value / len(y_true)

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

    def run_test(self, test_name, y_true, y_pred):
        print(test_name)
        print("========")

        TP, TN, FP, FN = self.confusion_matrix(y_true, y_pred)
        print(f"TP={TP}, TN={TN}, FP={FP}, FN={FN}")

        accuracy_result = self.accuracy(y_true, y_pred)
        print(f"Accuracy: {accuracy_result:.1f}")

        precision_value = self.precision(y_true, y_pred)
        print(f"Precision: {precision_value:.1f}")

        recall_value = self.recall(y_true, y_pred)
        print(f"Recall: {recall_value:.1f}")

        f1_value = self.f1_score(y_true, y_pred)
        print(f"F1 Score: {f1_value:.1f}")
        print("\n")


# --- Run Tests ---
ev = EvaluationTests()

ev.run_test(
    test_name = "Test Case 1 - Perfect Classification",
    y_true = [1, 0, 1, 0],
    y_pred = [1, 0, 1, 0]
)

ev.run_test(
    test_name = "Test Case 2 - All Wrong",
    y_true = [1, 1, 0, 0],
    y_pred = [0, 0, 1, 1]
)

ev.run_test(
    test_name = "Test Case 3 - No Predicted Positives",
    y_true = [1, 0, 1, 0],
    y_pred = [0, 0, 0, 0]
)

ev.run_test(
    test_name = "Test Case 4 - No Actual Positives",
    y_true = [0, 0, 0, 0],
    y_pred = [1, 0, 1, 0]
)