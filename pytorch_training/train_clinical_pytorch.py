import pandas as pd
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # Load data
    X_train = pd.read_pickle("X_train_c.pkl")
    y_train = pd.read_pickle("y_train_c.pkl")
    X_test = pd.read_pickle("X_test_c.pkl")
    y_test = pd.read_pickle("y_test_c.pkl")

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
    y_train_tensor = torch.tensor(y_train.values.astype(np.int64))
    X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
    y_test_tensor = torch.tensor(y_test.values.astype(np.int64))

    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 200), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        model = nn.Sequential(
            nn.Linear(185, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.2),
            nn.Linear(50, 3),
            nn.Softmax(dim=1)
        )


        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = loss_function(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)
            total = predicted.size(0)
            correct = (predicted == y_test).sum().item()
            acc.append(correct / total)

        cr = classification_report(y_test_tensor.numpy(), predicted.numpy(), output_dict=True)
        precision.append(cr["macro avg"]["precision"])
        recall.append(cr["macro avg"]["recall"])
        f1.append(cr["macro avg"]["f1-score"])

    print("Avg accuracy:", np.mean(acc))
    print("Avg precision:", np.mean(precision))
    print("Avg recall:", np.mean(recall))
    print("Avg f1:", np.mean(f1))
    print("Std accuracy:", np.std(acc))
    print("Std precision:", np.std(precision))
    print("Std recall:", np.std(recall))
    print("Std f1:", np.std(f1))
    print(acc)
    print(precision)
    print(recall)
    print(f1)

if __name__ == '__main__':
    main()