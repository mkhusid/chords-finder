''' This module contains utility functions for the project. '''
import numpy as np
import matplotlib.pyplot as plt


def plot_roc(predictions_df, classes):
    '''
    Plot ROC curve for each class.
    Args:
        predictions_df: DataFrame - DataFrame with predictions.
        classes: list[int] - list of classes.
    '''

    # Convert predictions to Pandas for easy manipulation
    predictions_df = predictions_df.select("genre_id", "probability").toPandas()

    # Plot ROC for each class
    plt.figure(figsize=(10, 8))

    for class_id, class_name in enumerate(classes):
        # Extract true labels and predicted probabilities for the current class
        true_labels = (predictions_df["genre_id"] == class_id).astype(int)
        probabilities = predictions_df["probability"].apply(lambda x: x[class_id])

        # Sort probabilities and compute TPR/FPR at different thresholds
        thresholds = np.sort(probabilities)
        tpr = []
        fpr = []

        for threshold in thresholds:
            tp = ((probabilities >= threshold) & (true_labels == 1)).sum()
            fn = ((probabilities < threshold) & (true_labels == 1)).sum()
            fp = ((probabilities >= threshold) & (true_labels == 0)).sum()
            tn = ((probabilities < threshold) & (true_labels == 0)).sum()

            tpr.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
            fpr.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

        # Sort FPR and TPR in ascending order of FPR for AUC calculation
        fpr_tpr = sorted(zip(fpr, tpr), key=lambda x: x[0])
        sorted_fpr = [x[0] for x in fpr_tpr]
        sorted_tpr = [x[1] for x in fpr_tpr]

        # Plot the ROC curve
        plt.plot(fpr, tpr,
                 label=f"Class {class_name} (AUC: {np.trapz(sorted_tpr, sorted_fpr):.2f})")

    # Add plot details
    plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")  # Diagonal
    plt.title("ROC Curve for Each Class")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid()
    plt.show()
