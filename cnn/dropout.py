import torch 

from torch import nn

torch.manual_seed(42)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Linear(4, 4)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = self.dropout(x)
        return x


model = SimpleNet()

x = torch.ones(1, 4)
model.eval()

y = model(x)

print(y.requires_grad)
print(y.grad_fn)