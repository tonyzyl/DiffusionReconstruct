import torch

class LpLoss(object):
    #loss function with rel/abs Lp loss, modified from neuralop:
    #https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/losses/data_losses.py
    '''
    LpLoss: Lp loss function, return the relative loss by default
    Args:
        d: int, start dimension of the field. E,g., for shape like (b, c, h, w), d=2 (default 1)
        p: int, p in Lp norm, default 2
        reduce_dims: int or list of int, dimensions to reduce
        reductions: str or list of str, 'sum' or 'mean' 

    Call: (y_pred, y)
    '''
    def __init__(self, d=1, p=2, reduce_dims=0, reductions='sum'):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y):
        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def rel(self, x, y):

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.rel(y_pred, y)

class EDMLoss:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
    
    def __call__(self, y_pred, y, sigma, **kwargs):
        # (comment from diffuser) We are not doing weighting here because it tends result in numerical problems.
        # See: https://github.com/huggingface/diffusers/pull/7126#issuecomment-1968523051
        # There might be other alternatives for weighting as well:
        # https://github.com/huggingface/diffusers/pull/7126#discussion_r1505404686
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss = weight * ((y - y_pred) ** 2)
        return loss.mean()

class EDMLoss_reg:
    def __init__(self, sigma_data=0.5, reg_weight=0.001):
        self.sigma_data = sigma_data
        self.reg_weight = reg_weight    
    
    def __call__(self, y_pred, y, sigma, **kwargs):
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        squared_err = ((y - y_pred) ** 2)
        csv_reg = (torch.sum(y_pred, dim=[-2, -1], keepdim=True) - torch.sum(y, dim=[-2, -1], keepdim=True))**2
        loss = weight * squared_err + self.reg_weight * csv_reg
        return loss.mean()