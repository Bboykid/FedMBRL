import pandas as pd
import numpy as np

def process_dfs(df):
    """
    Process the DataFrame to create input-output pairs for training.
    The input is the original data, and the output is the data shifted by one time step.
    """
    # Remove the first row
    df = df.tail(len(df) - 1)

    # Create a copy of the DataFrame for the output (dfY)
    dfY = df.copy()
    
    # Drop the "actions" column from the output DataFrame
    dfY.drop(["actions"], axis=1, inplace=True)

    # Drop the first row from the output DataFrame
    dfY = dfY.drop(dfY.index[[0]])
    
    # Drop the last row from the input DataFrame
    df = df.drop(df.index[[len(df) - 1]])

    # Align the indices of input and output DataFrames
    dfY.index = df.index

    # Reset indices of both DataFrames
    df = df.reset_index(drop=True)
    dfY = dfY.reset_index(drop=True)
    
    return df, dfY

def process_dfs_diff(df):
    """
    Process the DataFrame to create input-output pairs for training.
    The input is the original data, and the output is the state change (next state - current state).
    """
    # Create a copy of the DataFrame for the output (dfY)
    dfY = df.copy()
    
    # Drop the "actions" column from the output DataFrame
    dfY.drop(["actions"], axis=1, inplace=True)

    # Calculate the state change (next state - current state)
    dfY = dfY.diff().dropna()

    # Drop the first row from the input DataFrame
    df = df.drop(df.index[[len(df) - 1]])

    # Align the indices of input and output DataFrames
    dfY.index = df.index

    # Reset indices of both DataFrames
    df = df.reset_index(drop=True)
    dfY = dfY.reset_index(drop=True)
    
    return df, dfY
    
    

# def process_dfs


def expand_action_column(data_X, action_column_name="actions"):
    """
    Expand the specified column containing numpy arrays and merge it with other columns.
    
    Parameters:
    data_X: pd.DataFrame - Input data
    action_column_name: str - Column name to be expanded

    Returns:
    pd.DataFrame - Data with expanded column
    """
    # Split the "actions" column into multiple columns
    actions = data_X[action_column_name].apply(pd.Series)
    
    # Rename the new columns
    actions.columns = [f"{action_column_name}_{i}" for i in range(actions.shape[1])]
    
    # Drop the original "actions" column and concatenate the expanded columns
    data_X_expanded = pd.concat([data_X.drop(columns=[action_column_name]), actions], axis=1)
    
    return data_X_expanded


# def prepare_training_data_action(data_X, action_column_name="actions"):
#     """
#     将 DataFrame 转换为符合神经网络训练要求的 NumPy 数组，并展开指定列中的数组。
    
#     参数:
#     data_X: pd.DataFrame - 输入数据
#     action_column_name: str - 需要展开的列名

#     返回:
#     np.ndarray - 转换后的训练数据
#     """
#     # 获取 actions 列中的数组并展开
#     actions_expanded = np.vstack(data_X[action_column_name].values)
    
#     # 获取除 actions 列外的其他列数据
#     other_features = data_X.drop(columns=[action_column_name]).values
    
#     # 合并所有数据
#     training_data = np.hstack((other_features, actions_expanded))
    
#     return training_data