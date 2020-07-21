import torch
import torch.nn.functional as F

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

class MultiClassFF(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MultiClassFF, self).__init__()
        '''
        Hidden size = list[]
        '''
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_class = num_class
        self.layer_size = len(self.hidden_size)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_size[0])])
        self.linears.extend([torch.nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(self.layer_size-1)])
        self.linears.append(torch.nn.Linear(hidden_size[self.layer_size-1], self.num_class))

        self.relu = torch.nn.ReLU()
        self.sorfmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        # print(self.linears)
        for i in range(self.layer_size+1):
            x = self.linears[i](x)
            if i != self.layer_size:
                x = self.relu(x)
        output = self.sorfmax(x)
        return output

class MultiClassFF_v2(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MultiClassFF_v2, self).__init__()
        '''
        Hidden size = list[]
        '''
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.num_class = num_class
        self.layer_size = len(self.hidden_size)
        self.linears = torch.nn.ModuleList([torch.nn.Linear(input_size, hidden_size[0])])
        self.linears.extend([torch.nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(self.layer_size-1)])
        self.linears.append(torch.nn.Linear(hidden_size[self.layer_size-1], self.num_class))

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        for i in range(self.layer_size+1):
            x = self.linears[i](x)
            x = self.relu(x)
        output = self.sigmoid(x)
        return output
