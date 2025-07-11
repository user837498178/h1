import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoformerConfig, AutoformerForPrediction
from accelerate import Accelerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_loader import load_data, create_time_features, create_sequences_for_transformer

def main():
    # --- 1. Configuration ---
    context_length = 96  # Use 2 days of past data
    prediction_length = 48 # Predict 1 day into the future
    
    # --- 2. Load and Prepare Data ---
    df = load_data()
    df_features = create_time_features(df.copy())
    feature_cols = ['hour', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear']
    
    # ---------- 按时间顺序切分 dataframe ----------
    split_idx = int(0.8 * len(df_features))
    train_df = df_features.iloc[:split_idx].copy()
    test_df  = df_features.iloc[split_idx:].copy()

    # ---------- 仅在训练段拟合归一化器 ----------
    scaler = MinMaxScaler()
    train_df['KWH_scaled'] = scaler.fit_transform(train_df[['KWH']])
    test_df['KWH_scaled']  = scaler.transform(test_df[['KWH']])

    # 合并回完整顺序 DataFrame 供后续滑窗
    df_features['KWH_scaled'] = pd.concat([train_df['KWH_scaled'], test_df['KWH_scaled']]).sort_index()
    
    # --- 序列长度需包含最大 lag ---
    lags_sequence = [1, 24, 48]
    input_sequence_length = context_length + max(lags_sequence)

    # Create sequences (past_values 包含 context_length + max_lag)
    X, y, X_time, y_time = create_sequences_for_transformer(
        df_features,
        'KWH_scaled',
        feature_cols,
        input_sequence_length,
        prediction_length
    )

    # ---------- 顺序切分样本 ----------
    split_seq_idx = int(0.8 * len(X))
    X_train_val, X_test = X[:split_seq_idx], X[split_seq_idx:]
    y_train_val, y_test = y[:split_seq_idx], y[split_seq_idx:]
    X_time_train_val, X_time_test = X_time[:split_seq_idx], X_time[split_seq_idx:]
    y_time_train_val, y_time_test = y_time[:split_seq_idx], y_time[split_seq_idx:]

    # Further split training into training and validation
    X_train, X_val, y_train, y_val, X_time_train, X_time_val, y_time_train, y_time_val = train_test_split(
        X_train_val, y_train_val, X_time_train_val, y_time_train_val, test_size=0.2, random_state=42, shuffle=False
    )

    # --- 3. Create PyTorch Datasets and DataLoaders ---
    def to_tensor(data):
        return torch.from_numpy(data).float()

    # Create mask for observed values (all 1s in this case, as we have no missing data)
    # 这里使用 bool 类型是符合 Autoformer 的接口要求
    train_observed_mask = torch.ones(X_train.shape, dtype=torch.bool)
    val_observed_mask = torch.ones(X_val.shape, dtype=torch.bool)
    test_observed_mask = torch.ones(X_test.shape, dtype=torch.bool)

    # Autoformer 期望 past_values/future_values 形状为 (batch, seq_len)，不包含额外特征维度
    train_dataset = TensorDataset(to_tensor(X_train), to_tensor(X_time_train), train_observed_mask, to_tensor(y_train), to_tensor(y_time_train))
    val_dataset = TensorDataset(to_tensor(X_val), to_tensor(X_time_val), val_observed_mask, to_tensor(y_val), to_tensor(y_time_val))
    test_dataset = TensorDataset(to_tensor(X_test), to_tensor(X_time_test), test_observed_mask, to_tensor(y_test), to_tensor(y_time_test))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # --- 4. Initialize Model ---
    config = AutoformerConfig(
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=1,
        num_time_features=len(feature_cols),
        lags_sequence=lags_sequence,  # 确保最大的 lag 不超过 context_length
        d_model=32,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        dropout=0.1,
        activation_function="gelu",
    )
    model = AutoformerForPrediction(config)

    # --- 5. Training ---
    accelerator = Accelerator()
    device = accelerator.device
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model, optimizer, train_loader, val_loader, test_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader
    )

    print("Training Autoformer model...")
    epochs = 3 # A few epochs for demonstration
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            past_values, past_time_features, past_observed_mask, future_values, future_time_features = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_values=future_values,
                future_time_features=future_time_features,
            )
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                past_values, past_time_features, past_observed_mask, future_values, future_time_features = [b.to(device) for b in batch]
                outputs = model(
                    past_values=past_values,
                    past_time_features=past_time_features,
                    past_observed_mask=past_observed_mask,
                    future_values=future_values,
                    future_time_features=future_time_features,
                )
                val_loss += outputs.loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss / len(val_loader):.4f}")


    # --- 6. Prediction and Evaluation ---
    print("Making predictions...")
    model.eval()
    predictions_scaled = []
    actuals_scaled = []

    with torch.no_grad():
        for batch in test_loader:
            past_values, past_time_features, past_observed_mask, future_values, future_time_features = [b.to(device) for b in batch]
            
            outputs = accelerator.unwrap_model(model).generate(
                past_values=past_values,
                past_time_features=past_time_features,
                past_observed_mask=past_observed_mask,
                future_time_features=future_time_features
            )
            
            # outputs.sequences 形状为 (batch, num_samples, prediction_length)
            preds = outputs.sequences.mean(dim=1)  # 取样本均值，得到 (batch, prediction_length)
            predictions_scaled.append(preds.cpu().numpy())
            actuals_scaled.append(future_values.cpu().numpy())

    predictions_scaled = np.concatenate(predictions_scaled, axis=0)
    actuals_scaled = np.concatenate(actuals_scaled, axis=0)

    # Inverse transform and calculate RMSE
    # Note: We reshape to 2D for the scaler
    predictions = scaler.inverse_transform(predictions_scaled.reshape(-1, 1)).reshape(predictions_scaled.shape)
    actuals = scaler.inverse_transform(actuals_scaled.reshape(-1, 1)).reshape(actuals_scaled.shape)
    
    rmse = np.sqrt(np.mean((predictions - actuals)**2))
    print(f'Test RMSE: {rmse:.4f}')

    # --- 7. Plotting ---
    # Plot a sample from the test set
    plt.figure(figsize=(15, 7))
    plt.plot(actuals[0], label='Actual Consumption')
    plt.plot(predictions[0], label='Autoformer Predictions', linestyle='--')
    plt.title('Autoformer Model - Electricity Consumption Prediction (Sample)')
    plt.xlabel(f'Time steps (intervals of 30min) into the future (total {prediction_length})')
    plt.ylabel('KWH')
    plt.legend()
    plt.savefig('autoformer_predictions.png')
    print("Plot saved to autoformer_predictions.png")

if __name__ == '__main__':
    main() 