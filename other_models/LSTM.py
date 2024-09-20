import torch
import torch.nn as nn
import torch.nn.functional as F
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim=100,  # embedding_dim = 4
                 num_classes=2,
                 hidden_size=32,
                 lstm_layer_num=2,
                 dropout_rate=0.2,
                 fc_dim=100):

        super(BiLSTM, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.LSTM_layer_num = lstm_layer_num
        self.dropout = dropout_rate
        self.fc_dim = fc_dim

        # Bi-LSTM
        self.bilstm = nn.LSTM(input_size=embedding_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.LSTM_layer_num,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)

        # Fully connected layer
        self.d1 = nn.Linear(self.hidden_size*2, self.fc_dim)
        self.d2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, data):
        x = data[0]  # [batch, seq_len, feature]
        # Bi-LSTM
        lstm_out, _ = self.bilstm(x)
        x = lstm_out[:, -1, :]  # CNN-LSTM
        # Final layer
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        out = F.softmax(x, dim=1)
        return out