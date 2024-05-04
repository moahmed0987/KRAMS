import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def check_accuracy(target_labels, output_labels):
    if len(target_labels) != len(output_labels):
        print("Sample text and output text lengths do not match.")
        return 0
    correct = 0
    for i in range(len(target_labels)):
        if output_labels[i] == target_labels[i]:
            correct += 1
    print("Accuracy:", (correct / len(target_labels)) * 100, "%")
    return (correct / len(target_labels)) * 100

def classification_report(target_labels, output_labels):
    return metrics.classification_report(target_labels, output_labels)

def confusion_matrix(target_labels, output_labels, confusion_matrix_title):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(confusion_matrix_title)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    cmd = metrics.ConfusionMatrixDisplay.from_predictions(target_labels, output_labels, ax=ax)
    return cmd