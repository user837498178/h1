import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from data_loader import load_data, create_sequences_for_transformer


class HybridCNNTransformer(nn.Module):
    """结合 CNN 编码器和 Transformer 编码器块的混合模型。
    """

    def __init__(
        self,
        input_channels: int = 1,
        sequence_length: int = 96,
        prediction_length: int = 48,
        d_model: int = 512,
        n_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # ----- CNN 编码器 -----
        # 两个一维卷积层，将时间分辨率降低 2 倍
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2)
        self.norm2 = nn.GroupNorm(4, 32)
        self.relu = nn.ReLU()

        # 卷积后长度变为 sequence_length // 2
        reduced_len = sequence_length // 2
        flattened_size = 32 * reduced_len

        # 将展平结果映射到 (num_tokens, d_model)，其中 num_tokens=7.
        self.num_tokens = 7
        self.linear_proj = nn.Linear(flattened_size, self.num_tokens * d_model)

        # ----- Transformer 编码器 -----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ----- 预测头 -----
        # 先对 token 做均值池化，再通过 MLP 输出 prediction_length 个值
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, prediction_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        参数
        ----------
        x : Tensor, 形状 (batch, seq_len, channels)
            归一化后的历史负荷序列。
        返回
        -------
        Tensor, 形状 (batch, prediction_length)
            未来预测区间的点预测值。
        """
        # CNN 期望输入形状为 (batch, channels, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        # 展平
        batch_size = x.size(0)
        x = x.reshape(batch_size, -1)
        x = self.linear_proj(x)  # (batch, num_tokens * d_model)
        x = x.view(batch_size, self.num_tokens, -1)  # (batch, num_tokens, d_model)

        # Transformer
        x = self.transformer(x)  # same shape
        # 对 token 做均值池化
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.dropout(x)
        out = self.fc_out(x)  # (batch, prediction_length)
        return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------- 数据准备 ----------------
    context_length = 96  # 输入序列长度（过去 2 天，每 30 分钟一次）
    prediction_length = 48  # 预测长度（1 天）

    df = load_data()
    target = "KWH"

    # 对目标列进行归一化
    scaler = MinMaxScaler()
    df[target] = scaler.fit_transform(df[[target]])

    X, y, _, _ = create_sequences_for_transformer(
        df, target, feature_cols=[], sequence_length=context_length, prediction_length=prediction_length
    )

    # 调整形状为 (samples, seq_len, channels)
    X = X[..., None]
    y = y  # 已经是 (samples, prediction_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # ---------------- 模型 ----------------
    model = HybridCNNTransformer(
        input_channels=1,
        sequence_length=context_length,
        prediction_length=prediction_length,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = 5  # 为演示设置较小的训练轮数

    print("Training Hybrid CNN-Transformer model…")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}")

    # ---------------- 评估 ----------------
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            preds = model(batch_x)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(batch_y.numpy())

    preds_scaled = np.concatenate(preds_list, axis=0)
    targets_scaled = np.concatenate(targets_list, axis=0)

    # 反归一化
    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).reshape(preds_scaled.shape)
    targets = scaler.inverse_transform(targets_scaled.reshape(-1, 1)).reshape(targets_scaled.shape)

    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    print(f"Test RMSE: {rmse:.4f}")

    # ---------------- 不确定性估计（MC Dropout） ----------------
    T = 30  # Monte Carlo 采样次数
    dropout_preds = []
    model.train()  # 切换到训练模式以激活 Dropout
    with torch.no_grad():
        for _ in range(T):
            batch_preds = []
            for batch_x, _ in test_loader:
                batch_x = batch_x.to(device)
                batch_preds.append(model(batch_x).cpu().numpy())
            dropout_preds.append(np.concatenate(batch_preds, axis=0))

    dropout_preds = np.stack(dropout_preds, axis=0)  # (T, samples, pred_len)
    mean_preds = dropout_preds.mean(axis=0)
    var_preds = dropout_preds.var(axis=0)

    # 对平均预测结果进行反归一化以便绘图
    mean_preds_inv = scaler.inverse_transform(mean_preds.reshape(-1, 1)).reshape(mean_preds.shape)

    # 绘制
    plt.figure(figsize=(15, 6))
    plt.plot(targets[0], label="Actual")
    plt.plot(mean_preds_inv[0], label="Prediction", linestyle="--")
    plt.fill_between(
        np.arange(prediction_length),
        mean_preds_inv[0] - 1.96 * np.sqrt(var_preds[0]),
        mean_preds_inv[0] + 1.96 * np.sqrt(var_preds[0]),
        color="orange",
        alpha=0.3,
        label="95% CI",
    )
    plt.title("Hybrid CNN-Transformer Forecast (Sample)")
    plt.xlabel("Time steps (30-min)")
    plt.ylabel("KWH")
    plt.legend()
    plt.savefig("hybrid_cnn_transformer_predictions.png")
    print("Plot saved to hybrid_cnn_transformer_predictions.png")


if __name__ == "__main__":
    main() 