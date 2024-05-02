import os
import random
import torch
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    X_train_ = pd.read_pickle("img_train.pkl")
    X_train_ = pd.DataFrame(X_train_)["img_array"]

    X_test_ = pd.read_pickle("img_test.pkl")
    X_test_ = pd.DataFrame(X_test_)["img_array"]

    y_train = pd.read_pickle("img_y_train.pkl")
    y_train = y_train["label"].astype(np.float32).values.flatten()

    y_test = pd.read_pickle("img_y_test.pkl")
    y_test = y_test["label"].astype(np.float32).values.flatten()

    y_test[y_test == 2] = -1
    y_test[y_test == 1] = 2
    y_test[y_test == -1] = 1

    y_train[y_train == 2] = -1
    y_train[y_train == 1] = 2
    y_train[y_train == -1] = 1

    X_train = torch.stack([torch.Tensor(i) for i in X_train_])
    X_test = torch.stack([torch.Tensor(i) for i in X_test_])

    acc = []
    f1 = []
    precision = []
    recall = []
    seeds = random.sample(range(1, 200), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        model = nn.Sequential(
            nn.Conv2d(3, 100, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.5),
            nn.Conv2d(100, 50, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(50 * 16 * 16, 3),
            nn.Softmax(dim=1)
        )

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()

        print(model)

        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = loss_function(outputs, y_train)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}/{50}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            total = predicted.size(0)
            correct = (predicted == y_test).sum().item()
            acc.append(correct / total)

            cr = classification_report(y_test, predicted.numpy(), output_dict=True)
            precision.append(cr["macro avg"]["precision"])
            recall.append(cr["macro avg"]["recall"])
            f1.append(cr["macro avg"]["f1-score"])

    print("Avg accuracy: " + str(np.array(acc).mean()))
    print("Avg precision: " + str(np.array(precision).mean()))
    print("Avg recall: " + str(np.array(recall).mean()))
    print("Avg f1: " + str(np.array(f1).mean()))
    print("Std accuracy: " + str(np.array(acc).std()))
    print("Std precision: " + str(np.array(precision).std()))
    print("Std recall: " + str(np.array(recall).std()))
    print("Std f1: " + str(np.array(f1).std()))
    print(acc)
    print(precision)
    print(recall)
    print(f1)

if __name__ == '__main__':
    main()