import numpy as np
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    # Load data
    X_train = pd.read_pickle("X_train_vcf.pkl")
    y_train = pd.read_pickle("y_train_vcf.pkl")
    X_test = pd.read_pickle("X_test_vcf.pkl")
    y_test = pd.read_pickle("y_test_vcf.pkl")

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
            nn.Linear(15965, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 3),
            nn.Softmax(dim=1)
        )


        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss()

        for epoch in range(50):
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

    print("Avg accuracy:", np.array(acc).mean())
    print("Avg precision:", np.array(precision).mean())
    print("Avg recall:", np.array(recall).mean())
    print("Avg f1:", np.array(f1).mean())
    print("Std accuracy:", np.array(acc).std())
    print("Std precision:", np.array(precision).std())
    print("Std recall:", np.array(recall).std())
    print("Std f1:", np.array(f1).std())
    print(acc)
    print(precision)
    print(recall)
    print(f1)

if __name__ == '__main__':
    main()