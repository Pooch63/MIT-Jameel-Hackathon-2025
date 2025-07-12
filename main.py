import pandas as pd
from process import process_dataframe
from train import train_model
from plot import plot_pd_series_describe
from tqdm import tqdm
import sys

args = sys.argv[1:]

PROCESS = 'process' in args
TRAIN = 'train' in args
ANALYZE = 'analyze' in args

def train_epipen(dataframe: pd.DataFrame):
    return train_model(dataframe, output='Flag_EpiPen', device='cpu', epochs=20, save_folder='model_output')
def train_allergen_type(dataframe: pd.DataFrame):
    return train_model(dataframe, output='Allergen_Type', epochs=200, device='cpu', output_type='mc', save_folder='model_output',
                       hidden_layer_sizes=[512, 256, 128, 64, 32], batch_size=64)

if __name__ == '__main__':
    dataframe = None
    if PROCESS:
        dataframe = pd.read_csv("hackathon_data.csv")

        process_dataframe(dataframe)
        dataframe.to_csv("processed_hackathon_data.csv", index=False)
    else:
        dataframe = pd.read_csv("processed_hackathon_data.csv")

    print("Dataframe loaded successfully")
    if TRAIN:
        if '--epipen-model' in args:
            train_epipen(dataframe)
        elif '--allergen-type-model' in args:
            train_allergen_type(dataframe)
    if ANALYZE:
        cols = dataframe.columns
        for index in tqdm(range(len(cols)), desc="Analyzing columns"):
            col = cols[index]
            plot_pd_series_describe(dataframe[col], save_path=f"data_analysis/{col}_describe.png", title=f"Summary of {col}")
