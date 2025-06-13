# driver.py

import os
import time
import random
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import KFold

from mnb_classification import MNBClassification
from mnb_evaluation import MNBEvaluation

# ------------------------
# Load stopwords.txt
# ------------------------
def load_stopwords(filepath):
    with open(filepath, 'r') as f:
        stopwords = set(line.strip() for line in f if line.strip())
    return stopwords

# ------------------------
# Load 20NG dataset
# - Each subfolder = class label
# - Each file = one document
# - Output: list of (Counter(words), label)
# ------------------------
def load_20ng_dataset(base_folder, stopwords):
    dataset = []
    class_labels = os.listdir(base_folder)

    for label in class_labels:
        class_folder = os.path.join(base_folder, label)
        if not os.path.isdir(class_folder):
            continue

        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            with open(file_path, 'r', errors='ignore') as f:
                words = []
                for line in f:
                    # Lowercase, remove stopwords, use alpha words only
                    words.extend([word.lower() for word in line.strip().split() if word.isalpha() and word.lower() not in stopwords])
                word_counts = Counter(words)
                dataset.append((word_counts, label))

    # Shuffle dataset to avoid ordering bias
    random.shuffle(dataset)
    return dataset

# ------------------------
# Run 5-fold cross validation
# For given vocab size M:
# - train model
# - compute accuracy
# - measure training/test time
# ------------------------
def run_cross_validation(dataset, M, stopwords):
    kf = KFold(n_splits=5)

    accuracies = []
    train_times = []
    test_times = []

    for train_index, test_index in kf.split(dataset):
        # Split data into training/test sets
        training_set = [dataset[i] for i in train_index]
        test_set = [dataset[i] for i in test_index]

        # Initialize MNBClassifier
        clf = MNBClassification(stopwords)

        # Feature selection step
        if M < float('inf'):
            clf.featureSelection(training_set, M)
        else:
            clf.selectedFeatures = set()  # Use full vocab (no feature selection)

        # ------------------------
        # Training phase
        # ------------------------
        start_train = time.time()
        clf.mnb_prob.computeWordProbability(training_set)
        clf.mnb_prob.computeClassProbability(training_set)
        end_train = time.time()

        train_time = end_train - start_train

        # ------------------------
        # Testing phase
        # ------------------------
        start_test = time.time()
        predictions = []
        for doc_words, _ in test_set:
            pred_label = clf.label(doc_words)
            predictions.append(pred_label)
        end_test = time.time()

        test_time = end_test - start_test

        # ------------------------
        # Accuracy computation
        # ------------------------
        evaluator = MNBEvaluation()
        accuracy = evaluator.accuracyMeasure(test_set, predictions)

        # Record results
        accuracies.append(accuracy)
        train_times.append(train_time)
        test_times.append(test_time)

    # Compute average results over all 5 folds
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_train_time = sum(train_times) / len(train_times)
    avg_test_time = sum(test_times) / len(test_times)

    return avg_accuracy, avg_train_time, avg_test_time

# ------------------------
# Main driver
# Loads data
# Runs CV for various M values
# Plots accuracy + time graphs
# ------------------------
def main():
    # Load stopwords
    stopwords = load_stopwords('stopwords.txt')

    # Load dataset (20NG)
    dataset = load_20ng_dataset('20NG', stopwords)

    print(f"Loaded {len(dataset)} documents.")

    # Vocabulary sizes to test
    vocab_sizes = [float('inf'), 6200, 12400, 18600, 24800]
    accuracy_results = []
    train_time_results = []
    test_time_results = []

    # Loop through each vocab size
    for M in vocab_sizes:
        print(f"\nRunning 5-fold CV with vocab size = {M if M != float('inf') else 'Full Vocabulary'}...")

        avg_acc, avg_train_t, avg_test_t = run_cross_validation(dataset, M, stopwords)

        print(f"Accuracy: {avg_acc:.4f}, Train Time: {avg_train_t:.2f}s, Test Time: {avg_test_t:.2f}s")

        # Save results
        accuracy_results.append(avg_acc)
        train_time_results.append(avg_train_t)
        test_time_results.append(avg_test_t)

    # ------------------------
    # Plot Accuracy vs Vocab Size
    # ------------------------
    plt.figure()
    x_labels = ['Full'] + [str(m) for m in vocab_sizes[1:]]
    plt.plot(x_labels, accuracy_results, marker='o')
    plt.title('Accuracy vs Vocabulary Size')
    plt.xlabel('Vocabulary Size (M)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('accuracy_vs_vocab_size.png')
    plt.show()

    # ------------------------
    # Plot Train/Test Time vs Vocab Size
    # ------------------------
    plt.figure()
    plt.plot(x_labels, train_time_results, marker='o', label='Training Time')
    plt.plot(x_labels, test_time_results, marker='o', label='Testing Time')
    plt.title('Training/Test Time vs Vocabulary Size')
    plt.xlabel('Vocabulary Size (M)')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig('time_vs_vocab_size.png')
    plt.show()

# Entry point
if __name__ == '__main__':
    main()
