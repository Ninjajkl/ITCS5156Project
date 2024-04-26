import torch
import torch.nn as nn
import torch.nn.functional as F
class LSTMModel(nn.Module): 
    def __init__(self, vocab_size, hidden_size, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm1 = nn.LSTM(vocab_size, hidden_size)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.Softmax(dim=1)

    #The reason I have the state_ins is so I can predict the next char in a sequence later
    #This isn't used during training
    def forward(self, input, lstm_1_state_h_in=None, lstm_1_state_c_in=None,
                      lstm_2_state_h_in=None, lstm_2_state_c_in=None,
                      lstm_3_state_h_in=None, lstm_3_state_c_in=None):
        #Get batch size from the 2nd element of the input (which is the batch size)
        batch_size = input.size(1)
        
        #Initialize hidden and cell states if not provided
        if lstm_1_state_h_in is None:
            lstm_1_state_h_in = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        if lstm_1_state_c_in is None:
            lstm_1_state_c_in = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        if lstm_2_state_h_in is None:
            lstm_2_state_h_in = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        if lstm_2_state_c_in is None:
            lstm_2_state_c_in = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        if lstm_3_state_h_in is None:
            lstm_3_state_h_in = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        if lstm_3_state_c_in is None:
            lstm_3_state_c_in = torch.zeros(1, batch_size, self.hidden_size).to(input.device)
        
        out, (lstm_1_state_h_out, lstm_1_state_c_out) = self.lstm1(input, (lstm_1_state_h_in, lstm_1_state_c_in))
        out = self.dropout(out)
        out, (lstm_2_state_h_out, lstm_2_state_c_out) = self.lstm2(out, (lstm_2_state_h_in, lstm_2_state_c_in))
        out = self.dropout(out)
        out, (lstm_3_state_h_out, lstm_3_state_c_out) = self.lstm3(out, (lstm_3_state_h_in, lstm_3_state_c_in))
        out = self.dropout(out)
        
        out = self.linear(out)
        out = self.softmax(out)
        return out, lstm_1_state_h_out, lstm_1_state_c_out, lstm_2_state_h_out, lstm_2_state_c_out, lstm_3_state_h_out, lstm_3_state_c_out