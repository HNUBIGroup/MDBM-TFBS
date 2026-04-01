import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Dataset.DataPreprocessing import DataPreprocessor


class Constructor:

    def __init__(self, model):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5)
        self.loss_function = nn.BCELoss()
        self.batch_size = 64
        self.epochs = 1

    def train_model(self, TrainLoader, ValidateLoader):
        path = os.path.abspath(os.curdir)
        best = 1
        for epoch in range(self.epochs):
            self.model.train()
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()
                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, label = data
                output = self.model(seq.to(self.device), shape.to(self.device))
                loss = self.loss_function(output, label.float().to(self.device))
                ProgressBar.set_postfix(loss=loss.item())
                loss.backward()
                self.optimizer.step()

            valid_loss = []

            self.model.eval()
            with torch.no_grad():
                for valid_seq, valid_shape, valid_labels in ValidateLoader:
                    valid_output = self.model(valid_seq.to(self.device), valid_shape.to(self.device))
                    valid_labels = valid_labels.float().to(self.device)
                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())
                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)
            if valid_loss_avg < best:
                best = valid_loss_avg
                model_name = path + '\\save_model\\' + self.data_name + '.pth'
        torch.save(self.model.state_dict(), model_name)

    def test_model(self, TestLoader):
        path = os.path.abspath(os.curdir)
        self.model.load_state_dict(
            torch.load(path + '\\save_model\\' + self.data_name + '.pth',
                       map_location=self.device, weights_only=True))
        predicted_value = []
        true_label = []
        self.model.eval()
        with torch.no_grad():
            for seq, shape, label in TestLoader:
                output = self.model(seq.to(self.device), shape.to(self.device))
                predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
                true_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
            return predicted_value, true_label

    def estimate_model(self, predicted_value, true_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=true_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=true_label)
        precision, recall, _ = precision_recall_curve(y_score=predicted_value, y_true=true_label)
        pr_auc = auc(recall, precision)
        return accuracy, roc_auc, pr_auc

    def run_model(self, dataset_name, ratio=0.8):
        Train_Validate_Set = DataPreprocessor(dataset_name, False)
        Test_Set = DataPreprocessor(dataset_name, True)
        self.data_name = dataset_name
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) - math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))
        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True, batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True, batch_size=self.batch_size, shuffle=False, num_workers=0)
        TestLoader = loader.DataLoader(dataset=Test_Set, batch_size=1, shuffle=False, num_workers=0)
        self.train_model(TrainLoader, ValidateLoader)
        predicted_value, true_label = self.test_model(TestLoader)
        accuracy, roc_auc, pr_auc = self.estimate_model(predicted_value, true_label)
        return accuracy, roc_auc, pr_auc