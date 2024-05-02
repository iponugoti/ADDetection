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
    X_train = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/X_train_c.pkl")
    y_train = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/y_train_c.pkl")
    X_test = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/X_test_c.pkl")
    y_test = pd.read_pickle("/Users/timothypyon/Desktop/DL_Project/ADDetection/preprocess_clinical/y_test_c.pkl")

    X_train_post = X_train.replace({True: 1, False: 0, np.NAN: 0})
    y_train_post = y_train.replace({True: 1, False: 0.0, np.NAN: 0})

    X_test_post = X_test.replace({True: 1, False: 0, np.NAN: 0})
    y_test_post = y_test.replace({True: 1, False: 0.0, np.NAN: 0})

    X_train_tensor = torch.tensor(X_train_post.values.astype(np.float32), requires_grad=False)
    y_train_tensor = torch.tensor(y_train_post.values.astype(np.float32), requires_grad=False)
    X_test_tensor = torch.tensor(X_test_post.values.astype(np.float32), requires_grad=False)
    y_test_tensor = torch.tensor(y_test_post.values.astype(np.float32), requires_grad=False)


    acc = []
    f1 = []
    precision = []
    recall = []
    
    seeds = random.sample(range(1, 1000), 5)
    for seed in seeds:
        reset_random_seeds(seed)
        model = nn.Sequential(
            nn.Linear(100, 84),
            nn.ReLU(),
            nn.BatchNorm1d(84),
            nn.Dropout(0.5),
            nn.Linear(84, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.2),
            nn.Linear(50, 3),
        )
        


        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        loss_function = nn.CrossEntropyLoss()
        # loss_function = nn.NLLLoss()

        for epoch in range(100):
            model.train(True)
            optimizer.zero_grad()            
            outputs = model(X_train_tensor)
            loss = loss_function(outputs, torch.flatten(y_train_tensor.long()))
            print(loss)
            # loss = loss_function(outputs.type(torch.LongTensor), torch.flatten(y_train_tensor.type(torch.LongTensor)))
            # loss = loss_function(outputs.type(torch.FloatTensor), torch.flatten(y_train_tensor.type(torch.FloatTensor)))

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            voutputs = model(X_test_tensor)
            _, predicted = torch.max(voutputs, 1)
            total = predicted.size(0)
            count = 0
            num_right = 0
            for x in predicted:
                if x == y_test_tensor[count]:
                    num_right+=1
                count+=1
            correct = num_right
            # correct = (predicted == y_test_tensor).sum().item()
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