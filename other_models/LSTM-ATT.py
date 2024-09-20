import torch
import torch.nn as nn
import torch.nn.functional as F
class CNNAttention(nn.Module):
    def __init__(self, embedding_dim=100,  # embedding_dim = 4
                 num_classes=2,
                 attention_dim=64,
                 dropout_rate=0.2,
                 cnn_dim=64,
                 cnn_layer_num=3,
                 cnn_kernel_size=8,
                 fc_dim=100):

        super(CNNAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.attention_dim = attention_dim
        self.dropout = dropout_rate
        self.cnn_dim = cnn_dim
        self.cnn_layer_num = cnn_layer_num
        self.cnn_kernel_size = cnn_kernel_size
        self.fc_dim = fc_dim

        # Fully connected layer
        self.d1 = nn.Linear(self.hidden_size * 2, self.fc_dim)
        self.d2 = nn.Linear(self.fc_dim, self.num_classes)

        # Attention Layer
        self.attention_fc = nn.Linear(self.hidden_size * 2, self.attention_dim)
        self.attention_out = nn.Linear(self.attention_dim, 1)

    def attention(self, output):
        attention_weights = torch.tanh(self.attention_fc(output))
        attention_weights = self.attention_out(attention_weights)
        attention_weights = F.softmax(attention_weights, dim=1)
        attention_output = torch.sum(attention_weights * output, dim=1)
        return attention_output

    def forward(self, data):
        x = data[0]  # [batch, seq_len, feature]
        lstm_out, _ = self.bilstm(x)
        x = self.attention(lstm_out)  # CNN-Attention

        # Final layer
        x = self.d1(x)
        x = F.relu(x)
        x = self.d2(x)
        out = F.softmax(x, dim=1)
        return out