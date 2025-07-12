import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from plot import plot_losses, plot_roc_curve, plot_multiclass_roc, plot_metrics
from tqdm import tqdm

def train(model, dataloader, optimizer, device, criterion):

    '''
    A function train on the entire dataset for one epoch .

    Args:
        model (torch.nn.Module): your model from before
        dataloader (torch.utils.data.DataLoader): DataLoader object for the train data
        optimizer (torch.optim.Optimizer(()): optimizer object to interface gradient calculation and optimization
        device (str): Your device (usually 'cuda:0' for your GPU)

    Returns:
        float: loss averaged over all the batches

    '''

    epoch_loss = []
    model.train() # Set model to training mode

    # Loop through the data samples (patients)
    for batch in dataloader:
        X, y = batch
        X = X.to(device)  # take our features
        y = y.to(device)  # take the label (sepsis vs not sepsis)

        # train your model on each batch here
        outputs = model(X)  # run the model with the patient features
        outputs = outputs.squeeze()

        loss = criterion(outputs, y) # fill in loss here
        epoch_loss.append(loss.item())

        # Adjust the weights, according to how good/bad our prediction was as determined by the loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return np.array(epoch_loss).mean()

def validate(model, dataloader, device, criterion, type: str, sensitivity=0.5):
    '''
    A function validate on the validation dataset for one epoch .

    Args:
        model (torch.nn.Module): your model for before
        dataloader (torch.utils.data.DataLoader): DataLoader object for the validation data
        device (str): Your device ('cuda:0' for GPU or 'cpu' for CPU)

    Returns:
        float: loss averaged over all the batches

    '''

    val_loss = []
    all_pred = []
    all_true = []
    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch
            X = X.to(device)
            y = y.to(device)

            # validate your model on each batch here
            outputs = model(X)
            if type != 'mc':
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, y)# fill in loss here
            val_loss.append(loss.item())

            if type == 'mc':
                preds = torch.argmax(outputs, dim=1)
            else:
                preds = outputs > sensitivity

            all_pred.append(preds)
            all_true.append(y)

    all_pred = np.hstack(all_pred)
    all_true = np.hstack(all_true)

    acc = accuracy_score(all_pred, all_true)
    precision = precision_score(all_pred, all_true, average=('binary' if type == 'bc' else 'macro'))
    sensitivity = recall_score(all_pred, all_true, average=('binary' if type == 'bc' else 'macro'))

    metrics = {
        "accuracy": acc,
        "precision": precision,
        "sensitivity": sensitivity
    }

    return np.array(val_loss).mean(), metrics

def train_loop(model, train_dataloader, val_dataloader, optimizer, device: str, criterion, epochs: int, output_type: str):
    val_loss_curve = []
    train_loss_curve = []
    other_metrics = defaultdict(list)

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        # Compute train your model on training data
        epoch_loss = train(model, train_dataloader, optimizer, device, criterion)

        # Validate your on validation data
        val_loss, metrics = validate(model, val_dataloader, device, criterion, type=output_type)

        # Record train and loss performance
        train_loss_curve.append(epoch_loss)
        val_loss_curve.append(val_loss)

        # Pretty print metrics for this epoch
        string = f"Epoch = {(epoch + 1) :<2}, loss = {round(epoch_loss,5) :<8}, val_loss = {round(val_loss,5) :<8}"
        for name, metric in metrics.items():
            string = string + f", {name:<11} = {round(metric,3):<5}"
            other_metrics[name].append(metric)

        print('\n' + string)

    return val_loss_curve, train_loss_curve, other_metrics

# Define the Neural Network model for sepsis prediction
class MLP(nn.Module): #In python, we can make something called a "Class". We will encapsulate all our model in here !
  # Type can equal 'bc' for binary classification, 'mc' for multi-class classification, or
  # 'regression' for regression tasks.
  def __init__(self,
               input_size, inner_layers, output_size,
               dropout_rate: float = 0.2, type = 'bc',
               leaky_relu_negative_slope: float = 0.01):
      super().__init__()

      # Here is where we define the neural network and the layers !
      # You can put as many or as few layers as you would like
      # The only thing is that the first argument for self.layer1 should be input_size,
      # and the last argument for your last_layer should be output_size.
      # Addtionally, there should be activation functions between each layer (like torch.nn.ReLU())

      # Fill in the torch.nn.Sequential based on the number of layers that you input. Make a model that follows the tiny_example above!
      # Try one hidden layer with 32 neurons

      self.dropout = nn.Dropout(p=dropout_rate)

      #### ANSWER HERE #####
      sizes = [input_size, *inner_layers, output_size]
      
      self.layers = []
      for index in range(1, len(sizes)):
        self.layers.append(torch.nn.Linear(sizes[index - 1], sizes[index]))
        # Use Sigmoid for output layer (since it's a classification task)
        # and LeakyReLU for everything else
        if index < len(sizes) - 1:
          self.layers.append(torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope))
        else:
          if type == 'bc': self.layers.append(torch.nn.Sigmoid())
          elif type == 'mc': self.layers.append(torch.nn.Softmax(dim=1))
          else: self.layers.append(torch.nn.ReLU())  # For regression tasks
          
      self.model = torch.nn.Sequential(*self.layers)

  def forward(self, x):
        x = self.dropout(x)
        out = self.model(x)
        
        return out

def train_model(
        df: pd.DataFrame,
        output: str | int,
        features: list[str | int] | None = None,
        validation_size: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float | None = 1e-5,
        Loss=None,
        device='cpu',
        epochs: int = 40,
        save_folder: str | None = None,
        # bc, mc or regression
        output_type: str = 'bc',
        hidden_layer_sizes: list[int] | str | None = None,
        random_state: int | None = None,
        batch_size: int = 25):
    if Loss == None:
        if output_type == 'bc':
            Loss = nn.BCELoss
        elif output_type == 'mc':
            Loss = nn.CrossEntropyLoss
        else:
            Loss = nn.MSELoss

    if features == None:
        X = df.drop(columns=output).to_numpy()
    else:
        X = df[features].to_numpy()
    y_series = df[output]
    y = y_series.to_numpy()
    output_dim = 1 if output_type == 'bc' else len(y_series.unique())

    if hidden_layer_sizes == None:
        hidden_layer_sizes = []
        features = X.shape[1]
        if features < 2:
            hidden_layer_sizes = [2]
        elif features < 5:
            hidden_layer_sizes = [features + 1]
        else:
            num = len(features) // 2
            while num > max(2, output_dim):
                hidden_layer_sizes.append(num)
                num //= 2
            print(f"Hidden layer sizes: {hidden_layer_sizes}")

    print(f"X shape: {X.shape}, Y shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size, random_state=random_state)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Add dropouts and batch normalization to the model
    # graph validation vs training loss every 100th batch
    # record those values, then create a value
    # do for accuracy
    # create a confusion matrix

    if random_state != None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    # Use Float32's because there is 0 need for super-precise decimals, and our values are well within
    # float range
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    if output_type == 'mc':
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
    elif output_type == 'bc':
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

    train_data = TensorDataset(X_train, y_train) # Here we load our data into a Pytorch Dataset, which is just a way to bundle the features with their corresponding labels
    test_data = TensorDataset(X_test, y_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # We also use something called a "DataLoader" -- this is useful for feeding our data in batches into the model instead of all at once.
    val_dataloader = DataLoader(test_data, batch_size=len(y_test), shuffle=True)

    # We instantiate our model
    model = MLP(X.shape[1], hidden_layer_sizes, output_dim, type=output_type).to(device)

    # define your optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0 if weight_decay == None else weight_decay)
    # Here we define our loss function L.
    criterion = Loss()

    val_loss_curve, train_loss_curve, other_metrics = train_loop(
        model, train_dataloader, val_dataloader, optimizer, device, criterion, epochs, output_type)

    plot_losses(val_loss_curve, train_loss_curve, epochs, save_path = f"{save_folder}/{output}_loss_curve.png" if save_folder else None)
    if output_type == 'bc':
        plot_roc_curve(model, test_data, device=device, title='EpiPen ROC Curve', save_path=f"{save_folder}/{output}_roc_curve.png" if save_folder else None)
    elif output_type == 'mc':
        plot_multiclass_roc(model, val_dataloader, len(y_series.unique()), device=device, title='Allergen Type ROC Curve', save_path=f"{save_folder}/{output}_roc_curve.png" if save_folder else None)

    # Print final metrics
    plot_metrics(other_metrics, save_path=f"{save_folder}/{output}_metrics.png" if save_folder else None)
    print(other_metrics)

    return model

