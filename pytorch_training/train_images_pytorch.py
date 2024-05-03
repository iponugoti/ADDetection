import os
import random
import torch
import numpy as np
import pandas as pd
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess data
    X_train = pd.read_pickle("ADDetection/pytorch_training/img_train.pkl")["img_array"].apply(lambda x: torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32)).tolist()
    X_test = pd.read_pickle("ADDetection/pytorch_training/img_test.pkl")["img_array"].apply(lambda x: torch.tensor(x.transpose((2, 0, 1)), dtype=torch.float32)).tolist()
    y_train = pd.read_pickle("ADDetection/pytorch_training/img_y_train.pkl")["label"].values.astype(int)
    y_test = pd.read_pickle("ADDetection/training/img_y_test.pkl")["label"].values.astype(int)

    # Data transformations and dataset creation
    X_train = torch.stack(X_train).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test = torch.stack(X_test).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Define model
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
        nn.Linear(50 * 16 * 16, 3)  # Adjust this calculation as necessary
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(50):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{50}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_test).float().mean()
        print(f"Test Accuracy: {acc.item()}")

if __name__ == '__main__':
    main()
