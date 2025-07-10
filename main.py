import pandas as pd
from process import process_dataframe
from train import train_model

PROCESS = False
TRAIN = True

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
        train_model(dataframe, output='Flag_EpiPen', device='cpu', epochs=40)