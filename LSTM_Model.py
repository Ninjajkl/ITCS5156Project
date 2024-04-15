import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMModel(nn.Module): 
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    #The reason I have the h0 and c0 is so I can predict the next char in a sequence later
    #This isn't used during training
    def forward(self, x, h0=None, c0=None):
        if h0 is None or c0 is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_().to(x.device)

        #We need to detach as we are doing truncated backpropagation through time (BPTT)
        #If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out)

        return out, (hn,cn)