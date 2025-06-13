# mnb_evaluation.py

class MNBEvaluation:
    def accuracyMeasure(self, test_set, predicted_labels):
        """
        Compute accuracy = (# correct predictions) / (total)
        """
        correct = 0
        total = len(test_set)

        for i, (_, true_label) in enumerate(test_set):
            if predicted_labels[i] == true_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy
