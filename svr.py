import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from data_loader import load_data, create_sequences

def main():
    # Load and preprocess data
    data = load_data()
    kwh_values = data['KWH'].values

    # 时间顺序切分
    split_idx = int(0.8 * len(kwh_values))
    train_values = kwh_values[:split_idx]
    test_values  = kwh_values[split_idx:]

    # 仅在训练集拟合归一化器
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_values.reshape(-1, 1))
    test_scaled  = scaler.transform(test_values.reshape(-1, 1))

    # Create sequences
    sequence_length = 48  # 24 hours of data (48 half-hour intervals)
    X_train_seq, y_train = create_sequences(train_scaled, sequence_length)
    X_test_seq,  y_test  = create_sequences(test_scaled, sequence_length)

    # Reshape for SVR
    X_train = X_train_seq.reshape(X_train_seq.shape[0], X_train_seq.shape[1])
    X_test  = X_test_seq.reshape(X_test_seq.shape[0], X_test_seq.shape[1])

    # Initialize and train the SVR model
    print("Training SVR model...")
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(X_train, y_train.ravel())

    # Make predictions
    print("Making predictions...")
    predictions_scaled = svr.predict(X_test)

    # Inverse transform the predictions and actual values
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1))
    y_test_inversed = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test_inversed, predictions))
    print(f'Test RMSE: {rmse:.4f}')

    # Plot the results for a sample of the test set
    plt.figure(figsize=(15, 6))
    plt.plot(y_test_inversed[:200], label='Actual Consumption')
    plt.plot(predictions[:200], label='SVR Predictions')
    plt.title('SVR Model - Electricity Consumption Prediction (Sample)')
    plt.xlabel('Time (half-hour intervals)')
    plt.ylabel('KWH')
    plt.legend()
    plt.savefig('svr_predictions.png')
    print("Plot saved to svr_predictions.png")

if __name__ == '__main__':
    main() 