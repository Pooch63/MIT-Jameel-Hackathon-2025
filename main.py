import pandas as pd
from process import process_dataframe
from train import train_model
from plot import plot_pd_series_describe
from tqdm import tqdm

PROCESS = False
TRAIN = True
ANALYZE = False

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
        train_model(dataframe, output='Flag_EpiPen', device='cpu', epochs=100, save_folder='model_output')
    if ANALYZE:
        cols = dataframe.columns
        for index in tqdm(range(len(cols)), desc="Analyzing columns"):
            col = cols[index]
            plot_pd_series_describe(dataframe[col], save_path=f"data_analysis/{col}_describe.png", title=f"Summary of {col}")