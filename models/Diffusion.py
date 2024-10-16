import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Denoise(nn.Module):
    def __init__(self, in_dims, out_dims, emb_size, conf, norm=False, dropout=0.5):
        super(Denoise, self).__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.in_dims = in_dims
        self.out_dims = out_dims
        self.time_emb_dim = emb_size
        self.norm = norm

        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

        out_dims_temp = self.out_dims

        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

        self.drop = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def forward(self, x, timesteps, mess_dropout=True):
        freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).to(self.device)
        temp = timesteps[:, None].float() * freqs[None]
        time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
        if self.time_emb_dim % 2:
            time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
        emb = self.emb_layer(time_emb)
        if self.norm:
            x = F.normalize(x)
        if mess_dropout:
            x = self.drop(x) #(2048,4771)
        h = torch.cat([x, emb], dim=-1)  # (2048,4781); (2048,32780)
        for i, layer in enumerate(self.in_layers): #layer:(4781,1000)
            h = layer(h)  
            h = torch.tanh(h)
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers)-1:
                h = torch.tanh(h)
        
        return h
        



class GaussianDiffusion(nn.Module):
    def __init__(self, noise_scale, noise_min, noise_max, steps, conf, beta_fixed=True):
        super(GaussianDiffusion, self).__init__()

        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.noise_scale = noise_scale
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.steps = steps

        self.edgeDropper = SpAdjDropEdge(self.conf['keepRate'])

        if noise_scale != 0:
            # 获取betas
            self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).to(device)
            if beta_fixed:
                self.betas[0] = 0.0001
            
            self.calculate_for_diffusion()
    
    def get_betas(self):
        # 获取beta的方式改变，原来：betas=torch.linspace(beta_start, beta_end, timesteps)
        start = self.noise_scale * self.noise_min
        end = self.noise_scale * self.noise_max
        variance = np.linspace(start, end, self.steps, dtype=np.float64)
        alpha_bar = 1 - variance
        betas = []
        betas.append(1 - alpha_bar[0])
        for i in range(1, self.steps):
            betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
        return np.array(betas)
    
    def calculate_for_diffusion(self):
        alphas = 1.0 - self.betas #Tensor(5,)
        self.alphas_cumprod = torch.cumprod(alphas, axis=0).to(self.device) # torch.cumprod()累乘
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(self.device), self.alphas_cumprod[:-1]]).to(self.device)
        self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).to(self.device)]).to(self.device)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod) #recip:分之一
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod -1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

    def p_sample(self, model, x_start, steps, sampling_noise=False):
        # model是训练好的（能够...）的模型
        if steps == 0:
            x_t = x_start
        else:
            t = torch.tensor([steps-1] * x_start.shape[0]).to(self.device)
            x_t = self.q_sample(x_start, t) #x_noisy；之前的x_t的获取方式：torch.randn(shape)

        indices = list(range(self.steps))[::-1] #倒序[t, t-1, ..., 0]

        # 下面是逆向过程，一步步的得到x_0
        for i in indices: #循环执行，依次得到x_{t-1}, x_{t-2}, ..., x_0
            t = torch.tensor([i] * x_t.shape[0]).to(self.device)
            # Use our model (noise predictor) to predict the mean
            model_mean, model_log_variance = self.p_mean_variance(model, x_t, t) # 得到均值和方差后，可以在当前的分布中采样，来获取数据
            # 下面的操作即为重采样
            if sampling_noise:
                noise = torch.randn_like(x_t)
                nonzero_mask = ((t!=0).float().view(-1, *([1]*(len(x_t.shape)-1)))) #新添加的
                x_t = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise # 采样得到一个新的输出结果，即x_{t-1}

            else:
                x_t = model_mean
        return x_t
    
    def q_sample(self, x_start, t, noise=None): # q_sample即为前向过程，利用x_0和t,得到x_t
        if noise is None:
            noise = torch.rand_like(x_start)
        return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start \
               + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    
    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        arr = arr.to(self.device)
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)
    
    def p_mean_variance(self, model, x, t):
        model_output = model(x, t, False)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = self._extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

        model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + 
                      self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
        
        return model_mean, model_log_variance
    
    # def training_losses(self, model, x_start, itmEmbeds, batch_index, model_feats):
    def training_losses(self, model, x_start, itmEmbeds, batch_index):
        batch_size = x_start.size(0)

        ts = torch.randint(0, self.steps, (batch_size,)).long().to(self.device) #构建T
        noise = torch.randn_like(x_start)
        if self.noise_scale != 0:
            x_t = self.q_sample(x_start, ts, noise) # 前向过程，直接求得x_t
        else:
            x_t = x_start
        
        model_output = model(x_t, ts) #bundel-x_t:(2048,4771) model_output; item-x_t:(2048,32770)

        mse = self.mean_flat((x_start - model_output) ** 2)

        weight = self.SNR(ts - 1) - self.SNR(ts)
        weight = torch.where((ts == 0), 1.0, weight)

        diff_loss = weight * mse

        # usr_model_embeds = torch.mm(model_output, model_feats)
        usr_model_embeds = torch.mm(model_output, itmEmbeds) #(2048,4771);(4771,128)
        usr_id_embeds = torch.mm(x_start, itmEmbeds) #(2048,4771);(4771,128)

        gc_loss = self.mean_flat((usr_model_embeds - usr_id_embeds) ** 2)

        return diff_loss, gc_loss
    
    def mean_flat(self, tensor):
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    
    def SNR(self, t):
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
    


class SpAdjDropEdge(nn.Module):
    def __init__(self, keepRate):
        super(SpAdjDropEdge, self).__init__()
        self.keepRate = keepRate
    
    def forward(self, adj):
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((torch.rand(edgeNum) + self.keepRate).floor()).type(torch.bool)

        newVals = vals[mask] / self.keepRate
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)
    

