import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output, _ = self.lstm(x.float())
        output = self.dropout(output)
        output = self.linear(output)
        output = self.softmax(output)
        return output