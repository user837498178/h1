import pandas as pd
import numpy as np

def create_time_features(df):
    """
    Creates time series features from a datetime index.
    """
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    return df

def load_data(
    file_path: str = "CC_LCL-FullData.csv",
    *,
    household_id: str | None = None,
    parse_dates: bool = True,
    aggregate: str = "mean",
) -> "pd.DataFrame":
    """Load the *full* Smart-Meter data set and return a cleaned DataFrame.
    Parameters
    ----------
    file_path : str, default "CC_LCL-FullData.csv"
        路径既可为绝对也可为相对；若为相对，将按当前工作目录→脚本同级
        两级路径进行查找。
    household_id : str | None, optional
        指定户号（``LCLid``）。若提供，则仅返回该用户的数据；
        若为 ``None``，则会将所有用户按 ``DateTime`` 进行 *{aggregate}* 聚合，
        形成单序列数据。
    parse_dates : bool, default True
        是否将 ``DateTime`` 解析为 ``datetime`` 并设为索引。
    aggregate : {"mean", "sum"}, default "mean"
        当 ``household_id`` 为 ``None`` 时，采用的聚合方式。
    """

    import os

    # 如果 file_path 是相对路径，优先尝试当前工作目录；如果不存在则尝试与脚本同级目录
    if not os.path.isabs(file_path) and not os.path.exists(file_path):
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidate_path = os.path.join(module_dir, file_path)
        if os.path.exists(candidate_path):
            file_path = candidate_path

    df = pd.read_csv(file_path)

    # 统一列名
    df.rename(columns={"KWH/hh (per half hour) ": "KWH"}, inplace=True)

    # 在聚合之前，先将 KWH 列安全转换为数值，避免包含空格/字符串导致后续 mean 失败
    df["KWH"] = pd.to_numeric(df["KWH"].astype(str).str.strip(), errors="coerce")

    # ---------------- 户号筛选 / 聚合 ----------------
    if household_id is not None:
        # 筛选指定户
        df = df[df["LCLid"] == household_id].copy()
    else:
        # 聚合所有户。若数据量巨大可在此处考虑分块读取。
        if aggregate not in {"mean", "sum"}:
            raise ValueError("aggregate must be 'mean' or 'sum'")
        agg_func = "mean" if aggregate == "mean" else "sum"
        df = (
            df.groupby("DateTime", as_index=False)["KWH"].agg(agg_func)
        )

    # ----------------------------------------------

    if parse_dates:
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df.set_index("DateTime", inplace=True)

    # 清洗缺失值
    df["KWH"] = pd.to_numeric(df["KWH"], errors="coerce")
    df["KWH"].interpolate(method="time", inplace=True)
    df["KWH"].fillna(method="bfill", inplace=True)

    return df

def create_sequences_for_transformer(data, target_col, feature_cols, sequence_length, prediction_length):
    """
    Creates sequences for Transformer-based models.
    """
    X, y = [], []
    X_time_features, y_time_features = [], []
    
    values = data[target_col].values
    features = data[feature_cols].values

    for i in range(len(data) - sequence_length - prediction_length + 1):
        # Input sequences
        X.append(values[i:(i + sequence_length)])
        X_time_features.append(features[i:(i + sequence_length)])
        
        # Output sequences
        y.append(values[i + sequence_length:i + sequence_length + prediction_length])
        y_time_features.append(features[i + sequence_length:i + sequence_length + prediction_length])

    return (np.array(X), np.array(y), 
            np.array(X_time_features), np.array(y_time_features)) 

def create_sequences(data: np.ndarray, sequence_length: int):
    """Generate input/label pairs for classic time-series models.

    给定 ``data``（形状 (N, features) 或 (N,)），按滑动窗口方式构造样本：

    X[i] = data[i : i + sequence_length]
    y[i] = data[i + sequence_length]

    经典 CNN / RNN / 传统 ML 模型通常只预测下一步，所以 ``y`` 形状为 (samples, features)。
    对于一维序列 ``features = 1`` 时，最终 ``X`` 形状 (samples, sequence_length, 1)，``y`` 形状 (samples, 1)。
    """
    # 确保是二维数组 (N, features)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i : i + sequence_length])
        y.append(data[i + sequence_length])

    return np.array(X), np.array(y) 