import torch
from torch import nn

rnn = nn.LSTM(10, 20, 1)
input = torch.randn(1, 3, 10)
h0 = torch.randn(1, 3, 20)
c0 = torch.randn(1, 3, 20)
output, (hn, cn) = rnn(input, (h0, c0))

#print(output.shape)
#print(hn.shape)
#print(cn.shape)

print(output[0][0])
print(hn[0][0])

