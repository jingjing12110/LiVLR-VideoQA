import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            # nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            # nn.BatchNorm1d(hid_dim),
            nn.LayerNorm(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None),
        )

    def forward(self, x):
        return self.classifier(x)


# open-ended answer
class OpenEndedClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_answers=1000, dropout=0.0):
        super(OpenEndedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            weight_norm(nn.Linear(input_dim, hidden_dim), dim=None),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, num_answers), dim=None),
            )

    def forward(self, x):
        return self.classifier(x)
