import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from data_loader import load_data, create_sequences

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        # We only want the output of the last time step
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and preprocess data
    data = load_data()
    kwh_values = data['KWH'].values  # 1D ndarray

    split_idx = int(0.8 * len(kwh_values))
    train_values = kwh_values[:split_idx]
    test_values  = kwh_values[split_idx:]

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))
    test_scaled  = scaler.transform(test_values.reshape(-1, 1))

    sequence_length = 48
    X_train, y_train = create_sequences(train_scaled, sequence_length)
    X_test,  y_test  = create_sequences(test_scaled, sequence_length)

    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=64)

    model = LSTMModel().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 10
    print("Training LSTM model...")
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
    plt.plot(predictions[:200], label='LSTM Predictions')
    plt.title('LSTM Model (PyTorch) - Electricity Consumption Prediction (Sample)')
    plt.xlabel('Time (half-hour intervals)')
    plt.ylabel('KWH')
    plt.legend()
    plt.savefig('lstm_predictions.png')
    print("Plot saved to lstm_predictions.png")


if __name__ == '__main__':
    main() 