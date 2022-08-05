import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def load_data(forder_dir):
    train_data = pd.read_csv(os.path.join(forder_dir, 'train_data.csv'))
    train_data = train_data["MT_320"].values

    test_data = pd.read_csv(os.path.join(forder_dir, 'test_data.csv'))
    test_data = test_data["MT_320"].values

    print(train_data.shape) 
    print(test_data.shape) 
    return train_data, test_data


def get_train_val_data(train_data, valid_data, scaler_path):
    # normalization
    scaler = MinMaxScaler()

    if len(train_data.shape) == 1:  # shape=(time_steps, )
        scaler = scaler.fit(np.expand_dims(train_data, axis=-1))
    elif len(train_data.shape) < 3:  # shape=(num_of_instance, input_dims)
        scaler = scaler.fit(train_data)
    else:  # shape=(num_of_instance, input_dims, time_steps)
        origin_shape = train_data.shape
        scaler = scaler.fit(np.transpose(train_data, (0, 2, 1)).reshape(-1, origin_shape[1]))

    scaled_data = []
    for data in [train_data, valid_data]:
        if len(train_data.shape) == 1:  # shape=(time_steps, )
            data = scaler.transform(np.expand_dims(data, axis=-1))
            data = data.flatten()
        elif len(data.shape) < 3:  # shape=(num_of_instance, input_dims)
            data = scaler.transform(data)
        else:  # shape=(num_of_instance, input_dims, time_steps)
            data = scaler.transform(np.transpose(data, (0, 2, 1)).reshape(-1, origin_shape[1]))
            data = np.transpose(data.reshape(-1, origin_shape[2], origin_shape[1]), (0, 2, 1))
        scaled_data.append(data)

    # save scaler
    print(f"Save MinMaxScaler in path: {scaler_path}")
    pickle.dump(scaler, open(scaler_path, 'wb'))
    return scaled_data


def get_test_data(test_data, scaler_path):
    # load scaler
    scaler = pickle.load(open(scaler_path, 'rb'))

    # normalization
    if len(test_data.shape) == 1:  # shape=(time_steps, )
        scaled_test_data = scaler.transform(np.expand_dims(test_data, axis=-1))
        scaled_test_data = scaled_test_data.flatten()
    elif len(test_data.shape) < 3:  # shape=(num_of_instance, input_dims)
        scaled_test_data = scaler.transform(test_data)
    else:  # shape=(num_of_instance, input_dims, time_steps)
        origin_shape = test_data.shape
        scaled_test_data = scaler.transform(np.transpose(test_data, (0, 2, 1)).reshape(-1, origin_shape[1]))
        scaled_test_data = np.transpose(scaled_test_data.reshape(-1, origin_shape[2], origin_shape[1]), (0, 2, 1))
    return scaled_test_data, scaler


def get_plot(result_df):
    # set number of subplots (2000개의 데이터를 한 subplot에 시각화)
    num_fig = len(result_df) // 2000 + int(len(result_df) % 2000 != 0)
    fig, ax = plt.subplots(num_fig, 1, figsize=(24, 6 * num_fig))
    ax = [ax] if num_fig == 1 else ax

    for i in range(num_fig):
        # set true/predicted values for each subplot
        true_data = result_df.iloc[i*2000:(i+1)*2000].loc[:, 'actual_value']
        pred_data = result_df.iloc[i*2000:(i+1)*2000].loc[:, 'predicted_value']

        # plot true/predicted values
        ax[i].plot(true_data.index, true_data.values, alpha=0.5, label='Actual')
        ax[i].plot(pred_data.index, pred_data.values, alpha=0.5, label='Predicted')

        # set range of x and y axis
        min_x = i * 2000 if num_fig > 1 else 0
        max_x = (i + 1) * 2000 if num_fig > 1 else len(result_df)
        min_y = min(result_df['actual_value'].min(), result_df['predicted_value'].min())
        max_y = max(result_df['actual_value'].max(), result_df['predicted_value'].max())

        ax[i].set_xlim(min_x, max_x)
        ax[i].set_ylim(min_y, max_y)
        ax[i].set_xlabel('Index')
        ax[i].set_ylabel('Value')
        ax[i].legend()
    
    plt.title('Actual Values vs. Predicted Values')
    plt.show()