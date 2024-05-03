import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve

def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class MultiModalModel(nn.Module):
    def __init__(self, mode):
        super(MultiModalModel, self).__init__()
        self.mode = mode
        self.clinical_model = self.create_clinical_model()
        self.img_model = self.create_img_model()
        
        if self.mode == 'MM_BA':
            self.attention = self.cross_modal_attention
        elif self.mode == 'MM_SA':
            self.attention = self.self_attention
        elif self.mode == 'MM_SA_BA':
            self.attention = self.self_cross_modal_attention
        else:
            self.attention = None
        
        self.fc = nn.Linear(150, 3)  # Adjust the input size based on attention mechanism

    def create_clinical_model(self, input_size=9):
       
            model = nn.Sequential(
                nn.Linear(9, 84),
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
            return model

    def create_img_model(self):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Conv2d(72, 100, kernel_size=3, stride=1, padding=1),  # Adjust padding to maintain spatial dimensions
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Adjust stride to halve the spatial dimensions
            nn.Dropout(0.5),
            nn.Conv2d(100, 50, kernel_size=3, stride=1, padding=1),  # Adjust padding to maintain spatial dimensions
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Adjust stride to halve the spatial dimensions
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(50 * 18 * 18, 3)  # Adjust this calculation as necessary
        ).to(device)
        return model


    def cross_modal_attention(self, x, y):
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        a1 = nn.MultiheadAttention(50, num_heads=4)(x, y)[0]
        a2 = nn.MultiheadAttention(50, num_heads=4)(y, x)[0]
        a1 = a1[:, 0, :]
        a2 = a2[:, 0, :]
        return torch.cat([a1, a2], dim=1)

    def self_attention(self, x):
        x = x.unsqueeze(1)
        attention = nn.MultiheadAttention(50, num_heads=4)(x, x)[0]
        attention = attention[:, 0, :]
        return attention

    def forward(self, clinical_input, img_input):
        clinical_output = self.clinical_model(clinical_input)
        img_output = self.img_model(img_input)
        
        if self.attention:
            attention_output = self.attention(img_output, clinical_output)
            merged_output = torch.cat([attention_output, img_output, clinical_output], dim=1)
        else:
            merged_output = torch.cat([img_output, clinical_output], dim=1)
            
        output = self.fc(merged_output)
        return output

def make_img(t_img):
    X_train_ = pd.read_pickle(t_img)
    X_train_ = pd.DataFrame(X_train_)["img_array"]
    X_train = []
    for i in range(len(X_train_)):
        X_train.append(X_train_.values[i])
    return np.array(X_train)

def train(mode, batch_size, epochs, learning_rate, seed):
    reset_random_seeds(seed)
    
    train_clinical = pd.read_csv("X_train_clinical.csv").drop("Unnamed: 0", axis=1).set_index('subject')
    test_clinical = pd.read_csv("X_test_clinical.csv").drop("Unnamed: 0", axis=1).set_index('subject')
    
    # train_clinical = torch.tensor(train_clinical.values, dtype=torch.float32)
    # test_clinical = torch.tensor(test_clinical.values, dtype=torch.float32)

    train_clinical_numeric = train_clinical.select_dtypes(include=[np.number])  # Select only numeric columns
    train_clinical_tensor = torch.tensor(train_clinical_numeric.values, dtype=torch.float32)

    test_clinical_numeric = test_clinical.select_dtypes(include=[np.number])  # Select only numeric columns
    test_clinical_tensor = torch.tensor(test_clinical_numeric.values, dtype=torch.float32)

    
    train_img = torch.tensor(make_img("X_train_img.pkl"), dtype=torch.float32) / 255.0
    test_img = torch.tensor(make_img("X_test_img.pkl"), dtype=torch.float32) / 255.0
    
    train_label = pd.read_csv("y_train.csv").drop("Unnamed: 0", axis=1).values.astype("int").flatten()
    test_label = pd.read_csv("y_test.csv").drop("Unnamed: 0", axis=1).values.astype("int").flatten()
    
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_label), y=train_label)
    d_class_weights = dict(enumerate(class_weights))
    
    model = MultiModalModel(mode)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(weight=torch.Tensor(list(d_class_weights.values())))
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_clinical_tensor, train_img)
        loss = loss_function(outputs, torch.tensor(train_label, dtype=torch.long))
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(test_clinical_tensor, test_img)
        _, predicted = torch.max(outputs, 1)
        total = predicted.size(0)
        correct = (predicted == torch.tensor(test_label, dtype=torch.long)).sum().item()
        acc = correct / total

        cr = classification_report(test_label, predicted.numpy(), output_dict=True)

    print("Classification Report:")
    print(classification_report(test_label, predicted.numpy()))
    
    print("Test Accuracy:", acc)
    
    return acc, batch_size, learning_rate, epochs, seed

if __name__ == "__main__":
    m_a = {}
    types = ['MM_SA', 'MM_BA', 'MM_SA_BA', 'None']
    for t in types:
        seeds = random.sample(range(1, 200), 5)
        for s in seeds:
            acc, bs_, lr_, e_, seed = train(t, 32, 50, 0.001, s)
            m_a[acc] = (t, acc, bs_, lr_, e_, seed)
        print(m_a)
        print('-' * 55)
        max_acc = max(m_a, key=float)
    print("Highest accuracy of:", max_acc, "with parameters:", m_a[max_acc])
