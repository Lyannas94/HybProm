import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, embedding_dim=100,  # embedding_dim = 4
                 num_classes=2,
                 dropout_rate=0.2,
                 cnn_dim=64,
                 cnn_layer_num=3,
                 cnn_kernel_size=8,
                 fc_dim=100):

        super(CNN, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.dropout = dropout_rate
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim

        # CNN
        self.cnn_layers = nn.ModuleList()
        for i in range(self.cnn_layer_num):
            in_channels = self.embedding_dim if i == 0 else int(
                self.cnn_dim / (2 ** (i - 1)))  # embedding_dim/64, 64/32, 32/16
            out_channels = int(self.cnn_dim / (2 ** i))  # 64/32, 32/16
            self.cnn_layers.append(nn.Conv1d(in_channels=in_channels,
                                             out_channels=out_channels,
                                             kernel_size=self.cnn_kernel_size))
        final_out_channels = int(self.cnn_dim / (2 ** (self.cnn_layer_num - 1)))
        self.maxpool_layer = nn.MaxPool1d(kernel_size=2)

        # Fully connected layer
        self.d1 = nn.Linear(30*final_out_channels, self.fc_dim)
        self.d2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, data):
        x = data[0]  # [batch, seq_len, feature]
        x = x.transpose(1, 2)  # [batch, feature, seq_len]
        # CNN
        for i in range(self.cnn_layer_num):
            x = self.cnn_layers[i](x)
            x = self.maxpool_layer(x)
            x = F.relu(x)
            if not i == 0:
                x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer
        x = x.flatten(start_dim=1)
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        out = F.softmax(x, dim=1)
        return out
