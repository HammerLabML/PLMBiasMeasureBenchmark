import numpy as np
import math
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

class CLFHead(torch.nn.Module):
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    
class SimpleCLFHead(torch.nn.Module): # this copies the BertForSequenceClassificationHead
    
    def __init__(self, input_size: int, output_size: int, dropout_prob=0.1):
        super().__init__()
        self.input_size = input_size
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.classifier = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.classifier(x)
        x = self.softmax(x)
        return x
    
class MLMHead(torch.nn.Module):
    def __init__(self, input_size: int, pretrained_head: torch.nn.Module):
        super().__init__()
        self.input_size = input_size
        self.base_head = pretrained_head
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.base_head(x)
        x = self.softmax(x)
        return x

# TODO: also want sklearn clf?
class CustomModel():

    def __init__(self, parameters: dict, model: torch.nn.Module):
        self.model = model
        self.batch_size = parameters['batch_size']
        self.criterion = parameters['criterion']
        self.lr = parameters['lr']
        self.optimizer = parameters['optimizer'](params=model.parameters(), lr=self.lr)
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def fit(self, X, y, epochs=2):        
        dataset = TensorDataset(torch.tensor(X), F.one_hot(torch.tensor(y)).float())
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.train()
        for epoch in range(epochs):
            loop = tqdm(loader, leave=True)
            epoch_loss = 0
            for batch in loop:
                self.optimizer.zero_grad()
                
                if torch.cuda.is_available():
                    X = batch[0].to('cuda')
                    y = batch[1].to('cuda')
                
                pred = self.model(X)
                loss = self.criterion(pred, y)
                loss.backward()

                self.optimizer.step()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
                
                loss = loss.detach().item()
                epoch_loss += loss

                if torch.cuda.is_available():
                    pred.to('cpu')
                    X = X.to('cpu')
                    y = y.to('cpu')
                del pred
                del X
                del y

        self.model.eval()
        torch.cuda.empty_cache()     
    
    def predict(self, X):
        dataset = TensorDataset(torch.tensor(X))
        loader = DataLoader(dataset, batch_size=self.batch_size)
        predictions = []
        
        loop = tqdm(loader, leave=True)
        for batch in loop:
            if torch.cuda.is_available():
                X = batch[0].to('cuda')

            pred = self.model(X)

            if torch.cuda.is_available():
                pred = pred.to('cpu')
                X = X.to('cpu')
                
            pred = pred.detach().numpy()
            predictions.append(pred)
            
            del X

        torch.cuda.empty_cache()
        return np.vstack(predictions)
    
