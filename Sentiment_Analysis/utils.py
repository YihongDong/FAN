import torch.nn as nn

def view_params(model) -> str:
    res = "Total parameters: " + str(sum(p.numel() for p in model.parameters())/1e6) + ' M'
    return res


class Pooling(nn.Module):
    def __init__(self, pool_type='mean'):
        super(Pooling, self).__init__()
        self.pool_type = pool_type
        if pool_type == 'mean':
            self.pool = nn.AdaptiveAvgPool1d(1)  
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool1d(1)  
        else:
            raise ValueError("pool_type must be either 'mean' or 'max'.")

    def forward(self, x):
        x = x.transpose(1, 2)  
        pooled = self.pool(x)  
        return pooled.squeeze(2) 