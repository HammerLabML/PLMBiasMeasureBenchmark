import numpy as np
import math
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from sklearn.decomposition import PCA
from embedding import BertHuggingface
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


optimizer = {'RMSprop': torch.optim.RMSprop, 'Adam': torch.optim.Adam}
criterions = {'BCEWithLogitsLoss': torch.nn.BCEWithLogitsLoss, 'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss}

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
        dtype = emb.dtype
        print("Debias: predict with k="+str(k))
        X = self.normalize(emb)
        debiased = [self.dropspace(X[i,:], self.pca.components_[:k]) for i in range(X.shape[0])]
        return self.normalize(np.asarray(debiased, dtype=dtype))
    

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
        assert parameters['optimizer'] in optimizer.keys(), "optimizer "+parameters['optimizer']+" not in the list"
        assert parameters['criterion'] in criterions.keys(), "criterion "+parameters['criterion']+" not in the list"
        self.model = model
        self.batch_size = parameters['batch_size']
        self.criterion = criterions[parameters['criterion']]()
        self.lr = parameters['lr']
        self.optimizer = optimizer[parameters['optimizer']](params=self.model.parameters(), lr=self.lr)
        
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def fit(self, X, y, epochs=2, weights=None):
        if weights is not None:
            dataset = TensorDataset(torch.tensor(X), torch.tensor(y), torch.tensor(weights))
        else:
            dataset = TensorDataset(torch.tensor(X), torch.tensor(y))#F.one_hot(torch.tensor(y)).float())
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
                    if weights is not None:
                        w = batch[2].to('cuda')
                
                pred = self.model(X)
                
                loss = self.criterion(pred, y)
                if weights is not None:
                    loss = loss * w
                    loss = loss.mean()
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
                if weights is not None:
                    w = w.to('cpu')
                    del w

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
    
    def __init__(self, parameters: dict, head: torch.nn.Module, debias = False):
        self.clf = CustomModel(parameters, head)
        self.debiaser = None
        self.debias_k = None
        if debias:
            self.debiaser = Debias()
            self.debias_k = parameters['debias_k']
        # threshold for classification:
        self.theta = 0.5
        
    def upsample_defining_embeddings(self, emb_per_group):
        upsampled = []
        max_len = 0
        for emb_list in emb_per_group:
            if len(emb_list) > max_len:
                max_len = len(emb_list)
        
        
        for emb_list in emb_per_group:
            new_list = emb_list.copy()
            while len(new_list) < max_len:
                add_sample = random.choice(emb_list)
                new_list.append(add_sample)
            upsampled.append(new_list)
        return upsampled
        
    def fit(self, emb: np.ndarray, y: np.ndarray, group_label: list = None, epochs=2, optimize_theta=False, weights=None):        
        if self.debiaser is not None and group_label is not None:
            print("fit and apply debiasing...")
            emb_per_group = []
            n_groups = max(group_label)+1
            for group in range(n_groups):
                emb_per_group.append([e for i, e in enumerate(emb) if group_label[i] == group])
            
            # upsample to have equal sized groups
            emb_per_group = self.upsample_defining_embeddings(emb_per_group)
            
            self.debiaser.fit(emb_per_group)
            emb = self.debiaser.predict(emb, self.debias_k)
        
        print("fit clf head...")
        if optimize_theta:
            if weights is not None:
                emb_train, emb_val, y_train, y_val, w_train, w_val = train_test_split(emb, y, weights, test_size=0.1, random_state=0)
            else:
                w_train = None
                emb_train, emb_val, y_train, y_val = train_test_split(emb, y, test_size=0.1, random_state=0)
            self.clf.fit(emb_train, y_train, epochs=epochs, weights=w_train)
        
            print("optimize classification threshold...")
            pred = self.clf.predict(emb_val)
            best_score = 0
            for theta in np.arange(0.2, 1.0, 0.05):
                y_pred = (np.array(pred) >= theta).astype(int)
                score = accuracy_score(y_val, y_pred)
                if score > best_score:
                    self.theta = theta
                    best_score = score
            print("use theta="+str(self.theta)+" which achieved accurcay: "+str(best_score))
            
        else:
            print(type(emb))
            self.clf.fit(emb, y, epochs=epochs)
            self.theta = 0.5
    
    def predict(self, emb: np.ndarray):
        if self.debiaser is not None and self.debiaser.pca is not None:
            emb = self.debiaser.predict(emb, k=self.debias_k)
        
        pred = self.clf.predict(emb)
        y_pred = (np.array(pred) >= self.theta).astype(int)
        return y_pred