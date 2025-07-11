import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from data_loader import load_data, create_sequences

class CNNModel(nn.Module):
    def __init__(self, sequence_length=48, output_size=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        # Calculate the flattened size after convolution and pooling
        # After conv1: (N, 64, 48)
        # After pool: (N, 64, 24)
        flattened_size = 64 * (sequence_length // 2)
        self.fc1 = nn.Linear(flattened_size, 100)
        self.fc2 = nn.Linear(100, output_size)

    def forward(self, x):
        # Reshape for Conv1d: (N, C, L) where C is in_channels, L is sequence length
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- 数据加载 -----
    data = load_data()
    kwh_values = data['KWH'].values  # 1D ndarray

    # ----- 按时间顺序切分（80% 训练 / 20% 测试）-----
    split_idx = int(0.8 * len(kwh_values))
    train_values = kwh_values[:split_idx]
    test_values  = kwh_values[split_idx:]

    # ----- 仅在训练集上拟合归一化器，避免数据泄露 -----
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))
    test_scaled  = scaler.transform(test_values.reshape(-1, 1))

    # ----- 构造序列 -----
    sequence_length = 48
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test,  y_test  = create_sequences(test_scaled, sequence_length)

    # 转为张量
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

    model = CNNModel(sequence_length=sequence_length).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Training CNN model...")
    for i in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        if (i+1) % 1 == 0:
            print(f'Epoch {i+1}/{epochs}, Loss: {single_loss.item():.4f}')

    print("Making predictions...")
    model.eval()
    with torch.no_grad():
        test_predictions_scaled = model(X_test)

    test_predictions_scaled = test_predictions_scaled.cpu().numpy()

    predictions = scaler.inverse_transform(test_predictions_scaled)
    y_test_inversed = scaler.inverse_transform(y_test.cpu().numpy())

    rmse = np.sqrt(np.mean((predictions - y_test_inversed)**2))
    print(f'Test RMSE: {rmse:.4f}')

    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inversed[:200], label='Actual Consumption')
    plt.plot(predictions[:200], label='CNN Predictions')
    plt.title('1D-CNN Model (PyTorch) - Electricity Consumption Prediction (Sample)')
    plt.xlabel('Time (half-hour intervals)')
    plt.ylabel('KWH')
    plt.legend()
    plt.savefig('cnn_predictions.png')
    print("Plot saved to cnn_predictions.png")

if __name__ == '__main__':
    main() 