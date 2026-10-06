import torch
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

        return x / (1 +torch.exp(-1.702 * x))   # fast 

        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.arange(-8, 8, 0.1, requires_grad=True)
    y = GELU()
    values = y(x)
    yd = torch.autograd.grad(values.sum(), x)[0] # ***
    
    plt.plot(x.detach(), values.detach())
    plt.plot(x.detach(), yd.detach())
    plt.show()