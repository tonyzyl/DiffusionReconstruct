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

'''
class EDMLoss:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data
    
    def __call__(self, y_pred, y, sigma, **kwargs):
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        loss = weight * ((y - y_pred) ** 2)
        return loss.mean()
'''
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, gamma=5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, mask=None, out_channels=None):

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        #if self.opts.lsun:
            # use larger sigma for high-resolution datasets
        #    sigma *= 380. / 80.
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        #n = torch.randn_like(y) * sigma
        n = torch.zeros_like(y)
        if mask is not None:
            n[:, :out_channels] = torch.randn_like(y[:, :out_channels]) * sigma
            n[:, :out_channels] = n[:, :out_channels] * (1 - mask) 
        else:
            n = torch.randn_like(y) * sigma
        tmp_input = y + n
        D_yn = net(tmp_input, sigma.flatten(), labels)
        #print("y shape:", y.shape, "n shape:", n.shape)
        #D_yn = net(y + n, sigma, labels)

        target = y
        if mask is None:
            loss = weight * ((D_yn - target) ** 2)
        else:
            loss = weight * ((D_yn - target[:, :out_channels]) ** 2)
        return loss
#'''