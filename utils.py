import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from matplotlib.patches import Rectangle

scaler = StandardScaler()
criterion = nn.MSELoss()


def extract_samples_auc(df):
    x = df['Segment'].values
    x = torch.tensor(np.vstack(x)).double()
    y = torch.tensor(df['Label'].values)
    return x, y


def extract_samples(train_df, test_df, valid_df):
    X_train, y_train = extract_samples_auc(train_df)
    X_test,  y_test  = extract_samples_auc(test_df)
    X_valid, y_valid  = extract_samples_auc(valid_df)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

def divide_sets(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=24)
    return train_df, test_df, valid_df

def plot_loss(train_loss, val_loss_arr, lr):
    epochs = range(train_loss.shape[0])
    fig, ax1 = plt.subplots()

    # Plotting the train and validation errors
    ax1.plot(epochs, train_loss, 'b-', label='Train Error')
    ax1.plot(epochs, val_loss_arr, 'g-', label='Validation Error')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Error', color='b')
    ax1.tick_params('y', colors='b')
    ax1.legend(loc='upper left')

    # Creating a second y-axis for the scheduler
    ax2 = ax1.twinx()
    ax2.plot(epochs, lr, 'r--', label='Scheduler')
    ax2.set_ylabel('Learning Rate', color='r')
    ax2.tick_params('y', colors='r')
    ax2.legend(loc='upper right')

    plt.title('Training and Validation Error with lr Scheduler')
    plt.show()


def plot_results(true_label, predicted):
    plt.figure()
    plt.scatter(true_label, predicted, color='blue', label='Predicted vs. True')
    plt.plot(true_label.squeeze(), true_label.squeeze(), color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel('Time to AF termination [s]')
    plt.ylabel('Predicted Time [s]')
    threshold1 = 60
    threshold2 = 120

    # Add shaded regions
    thresholds = [60, 120]
    # colors = ['green', 'orange', 'red']  # Colors for the regions
    colors = ['green', 'red']  # Colors for the regions
    # for i, threshold in enumerate(thresholds):
    #     plt.axline((threshold, threshold), slope=1, color=colors[i], linestyle='--', label=f'Threshold {i + 1}')

    # Fill the regions between the thresholds
    rectangles = [
        Rectangle((0, 0), thresholds[1], thresholds[1], color=colors[0], alpha=0.3, label='Region 1 (<120)'),
        # Rectangle((thresholds[0], thresholds[0]), thresholds[1] - thresholds[0], thresholds[1] - thresholds[0],
        #           color=colors[1], alpha=0.3, label='Region 2 (60-120)'),
        Rectangle((thresholds[1], thresholds[1]), 240 - thresholds[1], 240 - thresholds[1], color=colors[1], alpha=0.3,
                  label='Region 3 (>120)')
    ]

    # Plot the rectangles
    for rectangle in rectangles:
        plt.gca().add_patch(rectangle)
    plt.title('Regression Results')
    plt.grid(True)
    plt.legend()
    plt.show()

def calc_confusion(true_labels, predicted_labels):
    true_labels = true_labels.squeeze().astype(int)
    predicted_labels = predicted_labels.squeeze().astype(int)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, pos_label=2)
    recall = recall_score(true_labels, predicted_labels, pos_label=2)
    f1 = f1_score(true_labels, predicted_labels, pos_label=2)
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(conf_matrix)


def plot_example(df):
    terminate = df.index[df['True_label'] == 0][40]
    non_terminate = df.index[df['True_label'] == 2][40]
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(df['Segment'].loc[terminate])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title("Terminating AF")

    plt.subplot(2, 1, 2)
    plt.plot(df['Segment'].loc[non_terminate])
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title("Non-terminating AF")
    plt.tight_layout()
    plt.show()