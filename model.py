import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, embedding_dim=100,  # embedding_dim = 4 
                 num_classes=2,
                 hidden_size=32,
                 lstm_layer_num=2,
                 attention_dim=64,
                 dropout_rate=0.2,
                 cnn_dim=64,
                 cnn_layer_num=3,
                 cnn_kernel_size=8,
                 fc_dim=100):

        super(CNNBiLSTMAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.LSTM_layer_num = lstm_layer_num
        self.attention_dim = attention_dim
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

        # Bi-LSTM
        self.bilstm = nn.LSTM(input_size=final_out_channels,
                              hidden_size=self.hidden_size,
                              num_layers=self.LSTM_layer_num,
                              batch_first=True,
                              dropout=self.dropout,
                              bidirectional=True)

        # Fully connected layer
        self.d1 = nn.Linear(self.hidden_size * 2, self.fc_dim)
        self.d2 = nn.Linear(self.fc_dim, self.num_classes)

        # Attention Layer
        self.attention_fc = nn.Linear(self.hidden_size * 2, self.attention_dim)
        self.attention_out = nn.Linear(self.attention_dim, 1)

    def attention(self, lstm_output):
        # lstm_output shape: [batch_size, seq_len, hidden_size * 2]
        attention_weights = torch.tanh(self.attention_fc(lstm_output))  # shape: [batch_size, seq_len, attention_dim]
        attention_weights = self.attention_out(attention_weights)  # shape: [batch_size, seq_len, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # shape: [batch_size, seq_len, 1]
        attention_output = torch.sum(attention_weights * lstm_output, dim=1)  # shape: [batch_size, hidden_size * 2]
        return attention_output

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
        x = x.transpose(1, 2)  # LSTM expects (batch, seq_len, features)
        # Bi-LSTM
        lstm_out, _ = self.bilstm(x)
        # Attention
        x = self.attention(lstm_out)  # CNN-LSTM-Attention

        # # t-SNE
        # features = x.detach().cpu().numpy()
        # tsne = TSNE(n_components=2)
        # features_2d = tsne.fit_transform(features)/10
        # custom_cmap = plt.cm.colors.ListedColormap([(201 / 255, 71 / 255, 55 / 255),(87 / 255, 130 / 255, 121 / 255)])
        # # 可视化降维后的特征
        # plt.figure()
        # plt.style.use("default")
        # scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=data[1].cpu().numpy(), alpha=0.6, s=5,
        #             cmap=custom_cmap)
        # plt.title('Escherichia coli K-12')
        # cbar = plt.colorbar(scatter, ticks=[0, 1])
        # cbar.ax.set_yticklabels(['neg', 'pos'])
        # plt.show()

        # Final layer
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        out = F.softmax(x, dim=1)

        return out
