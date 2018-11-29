import numpy as np
import pandas as pd


def data_converter(array_draw, dimension):
    pixel_two_dim_array = [[0] * dimension for _ in range(dimension)]
    for i, array in enumerate(array_draw):
        for i in range(len(array[0])):
            pixel_two_dim_array[array[0][i]][array[1][i]] = 1.0

    pixels_one_dim_array = np.array(pixel_two_dim_array).tolist()

    return pixels_one_dim_array


def concat_shuffle_dataframe(dataframes):
    df = pd.concat(dataframes).reset_index(drop=True)
    df = df.sample(frac=1).reset_index(drop=True)
    return df
