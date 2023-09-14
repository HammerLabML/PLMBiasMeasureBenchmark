import numpy as np
import math
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.decomposition import PCA
from embedding import BertHuggingfaceMLM

class Debias():

    def __init__(self):
        self.pca = None
        self.n_attributes = 1

    def sub_mean(self, pairs):
        means = np.mean(pairs, axis=0)
        for i in range(means.shape[0]):
            for j in range(pairs.shape[0]):
                pairs[j,i,:] = pairs[j,i,:] - means[i]
        return pairs

    def flatten(self, pairs):
        flattened = []
        for pair in pairs:
            for vec in pair:
                flattened.append(vec)
        return flattened

    def drop(self, u, v):
        return u - v * u.dot(v) / v.dot(v)

    def get_bias(self, u, V):
        norm_sqrd = np.sum(V * V, axis=-1)
        vecs = np.divide(V @ u, norm_sqrd)[:, None] * V
        subspace = np.sum(vecs, axis=0)
        return subspace

    def dropspace(self, u, V):
        return u - self.get_bias(u, V)

    def normalize(self, vectors: np.ndarray):
        norms = np.apply_along_axis(np.linalg.norm, 1, vectors)
        vectors = vectors / norms[:, np.newaxis]
        return np.asarray(vectors)

    def fit(self, embedding_tuples: list):
        self.n_attributes = len(embedding_tuples)
        
        embedding_tuples = self.normalize(np.asarray(embedding_tuples))
        assert embedding_tuples.shape[0] < embedding_tuples.shape[1], "the number of samples should be larger than the number of bias attributes"
        print("fit pca for debiasing")
        encoded_pairs = self.sub_mean(embedding_tuples)
        flattened = self.flatten(encoded_pairs)
        self.pca = PCA()
        self.pca.fit(flattened)
        print("explained variance of the first 20 principal components:")
        print(self.pca.explained_variance_ratio_[:20])

    def predict(self, emb, k=3):
        print("Debias: predict with k="+str(k))
        X = self.normalize(emb)
        debiased = [self.dropspace(X[i,:], self.pca.components_[:k]) for i in range(X.shape[0])]
        return self.normalize(np.asarray(debiased))
    

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

    
class CustomModel():

    def __init__(self, parameters: dict, model: torch.nn.Module):
        self.model = model
        self.batch_size = parameters['batch_size']
        self.criterion = parameters['criterion']
        self.lr = parameters['lr']
        self.optimizer = parameters['optimizer'](params=self.model.parameters(), lr=self.lr)
        
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
    

class DebiasPipeline():
    
    def __init__(self, parameters: dict, embedder: BertHuggingfaceMLM, head: torch.nn.Module, debias = False):
        self.clf = CustomModel(parameters, head)
        self.debiaser = None
        if debias:
            self.debiaser = Debias()
        self.debias_k = parameters['debias_k']
        self.embedder = embedder
        
    def fit(self, X: list, y: list, group_label: list = None, epochs=2):
        print("embed samples...")
        emb = self.embedder.embed(X)
        
        if self.debiaser is not None and group_label is not None:
            print("fit and apply debiasing...")
            emb_per_group = []
            n_groups = max(group_label)+1
            for group in range(n_groups):
                emb_per_group.append([e for i, e in enumerate(emb) if group_label[i] == group])
            self.debiaser.fit(emb_per_group)
            emb = self.debiaser.predict(emb, self.debias_k)
        
        print("fit clf head...")
        head.fit(emb, y, epochs=epochs)
    
    def predict(self, X: list):
        emb = self.embedder.embed(X)
        
        if self.debiaser is not None and self.debiaser.pca is not None:
            emb = self.debiaser.predict(emb, k=self.debias_k)
        
        pred = head.predict(emb)
        return pred