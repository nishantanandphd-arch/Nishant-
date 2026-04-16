import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import DataLoader, TensorDataset

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_enc = LabelEncoder()

    def fit_transform(self, df, label_col):
        df = df.drop_duplicates().fillna(0)

        y = self.label_enc.fit_transform(df[label_col])
        X = df.drop(columns=[label_col])

    
        X = X.select_dtypes(include=[np.number])

        X = self.scaler.fit_transform(X)
        return X, y


def select_top_features(X, k=16):
    scores = np.var(X, axis=0)
    idx = np.argsort(scores)[-k:]
    return idx

class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_model(model, X_train, y_train, X_val, y_val, epochs=80):
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True
    )

    class_counts = np.bincount(y_train.numpy())
    weights = 1. / class_counts
    weights = torch.tensor(weights, dtype=torch.float32)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            out = model(xb)
            loss = criterion(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred = model(X_val).argmax(1)
                acc = accuracy_score(y_val.numpy(), pred.numpy())

            print(f"Epoch {epoch} | Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")


def evaluate(model, X_test, y_test):
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model.eval()
    with torch.no_grad():
        pred = model(X_test).argmax(1)

    y_true = y_test.numpy()
    y_pred = pred.numpy()

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    specificity_list = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FP + FN)

        spec = TN / (TN + FP) if (TN + FP) != 0 else 0
        specificity_list.append(spec)

    specificity = np.mean(specificity_list)

    print("\n📊 Final Results:")
    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"Specificity  : {specificity:.4f}")


def run(csv_path, label_col):
    df = pd.read_csv(csv_path)

    prep = Preprocessor()
    X, y = prep.fit_transform(df, label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    idx = select_top_features(X_train, k=16)

    X_train = X_train[:, idx]
    X_val = X_val[:, idx]
    X_test = X_test[:, idx]

    model = SimpleModel(input_dim=16, num_classes=len(np.unique(y)))

    train_model(model, X_train, y_train, X_val, y_val, epochs=80)

    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    run(r"Enter your CSV path", "label")
    