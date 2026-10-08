import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("processed_dataset.csv")

df_human = df[df['label'] == 1]
df_bot = df[df['label'] == 0]
min_size = min(len(df_human), len(df_bot))

if min_size == 0:
    print("Ошибка: один из классов пуст!")
    exit()

df_human_balanced = df_human.sample(n=min_size, random_state=42)
df_bot_balanced = df_bot.sample(n=min_size, random_state=42)

df_balanced = pd.concat([df_human_balanced, df_bot_balanced], ignore_index=True)
df_balanced = df_balanced.sample(frac=1.0, random_state=42).reset_index(drop=True)

X = df_balanced[['mean_speed', 'std_speed', 'mean_accel', 'std_accel', 'straightness']].values
y = df_balanced['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

class AntiBotNet(nn.Module):
    def __init__(self, input_dim):
        super(AntiBotNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

model = AntiBotNet(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(250):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    predictions = model(X_test_t)
    predicted_classes = (predictions > 0.5).float()
    accuracy = (predicted_classes == y_test_t).float().mean()
    print(f"Сбалансированная точность (Balanced Accuracy): {accuracy.item() * 100:.2f}%")

torch.save(model.state_dict(), "antibot_model.pth")