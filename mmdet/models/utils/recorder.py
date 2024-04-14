import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist

class Single_Statistic_Recorder(torch.nn.Module):
    def __init__(self, channel, beta1=0.9, beta2=0.999, epislon=1e-8):
        super(Single_Statistic_Recorder, self).__init__()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epislon = epislon
        self.t = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.m = nn.Parameter(torch.zeros(channel), requires_grad=False)
        self.v = nn.Parameter(torch.zeros(channel), requires_grad=False)
        self.mu_hat = nn.Parameter(torch.zeros(channel), requires_grad=False)
        self.mu2_hat = nn.Parameter(torch.zeros(channel), requires_grad=False)
        self.var_hat = nn.Parameter(torch.zeros(channel), requires_grad=False)
        
        self.ev = nn.Parameter(torch.zeros(channel), requires_grad=False)
        self.avg_var_hat = nn.Parameter(torch.zeros(channel), requires_grad=False)

    def clear(self):
        self.t.data = self.t.data * 0
        self.m.data = self.m.data * 0
        self.v.data = self.v.data * 0
        self.mu_hat.data = self.mu_hat.data * 0
        self.mu2_hat.data = self.mu2_hat.data * 0
        self.var_hat.data = self.var_hat.data * 0
        self.ev.data = self.ev.data * 0
        self.avg_var_hat.data = self.avg_var_hat.data * 0

    def rejudge_noise(self, input_var):
        nan_mask = torch.isnan(input_var)
        real_idxs = torch.where(~nan_mask)[0]
        nan_idxs = torch.where(nan_mask)[0]
        min_val = input_var[real_idxs].min()
        input_var[nan_idxs] = min_val
        return input_var

    def forward(self, g, reduction='mean'):
        x = g
        with torch.no_grad():
            if reduction == 'mean':
                g2 = (g * g).mean(0)
                ev_g = g.var(0)
                g = g.mean(0)
            else:
                g2 = g * g
            
            self.t += 1
            self.m.data = self.beta1 * self.m.data + (1 - self.beta1) * g
            self.v.data = self.beta2 * self.v.data + (1 - self.beta2) * g2
            self.ev.data = self.beta2 * self.ev.data + (1 - self.beta2) * ev_g
            self.mu_hat.data = self.m.data / (1 - self.beta1 ** self.t)
            self.mu2_hat.data = self.v.data / (1 - self.beta2 ** self.t)
            self.avg_var_hat.data = self.ev.data / (1 - self.beta2 ** self.t)
            self.var_hat.data = self.mu2_hat.data - (self.mu_hat.data ** 2)
        
        return x


class Sync_Single_Statistic_Recorder(Single_Statistic_Recorder):
    def __init__(self, channel, beta1=0.9, beta2=0.999, epislon=1e-8):
        super(Sync_Single_Statistic_Recorder, self).__init__(channel, beta1, beta2, epislon)
    
    def forward(self, g, reduction='mean'):
        ans = g
        with torch.no_grad():
            x = g
            
            num_sample = x.shape[0] * torch.ones(1).type_as(x).to(device=x.device)
            dist.all_reduce(num_sample)
            
            if reduction == 'mean':
                g2 = (x * x).sum(0)
                dist.all_reduce(g2)
                g2 = g2 / torch.clamp(num_sample, min=1e-10)
                
                g = x.sum(0)
                dist.all_reduce(g)
                g = g / torch.clamp(num_sample, min=1e-10)
                
                ev_g = (x - g.view(1, -1))**2
                ev_g = ev_g.sum(0)
                dist.all_reduce(ev_g)
                ev_g = ev_g / torch.clamp(num_sample-1, min=1e-10)
            else:
                g2 = g * g
            
            self.t += 1
            self.m.data = self.beta1 * self.m.data + (1 - self.beta1) * g
            self.v.data = self.beta2 * self.v.data + (1 - self.beta2) * g2
            self.ev.data = self.beta2 * self.ev.data + (1 - self.beta2) * ev_g
            self.mu_hat.data = self.m.data / (1 - self.beta1 ** self.t)
            self.mu2_hat.data = self.v.data / (1 - self.beta2 ** self.t)
            self.avg_var_hat.data = self.ev.data / (1 - self.beta2 ** self.t)
            self.var_hat.data = self.mu2_hat.data - (self.mu_hat.data ** 2)
        
        return ans


class Statistics_Recorder(torch.nn.Module):
    def __init__(self, channel, beta1=0.9, beta2=0.999, epislon=1e-8, use_sync=False):
        super(Statistics_Recorder, self).__init__()
        self.epislon = epislon
        if use_sync:
            self.logit_recorder = Sync_Single_Statistic_Recorder(channel, beta1, beta2, epislon)
            self.score_recorder = Sync_Single_Statistic_Recorder(channel, beta1, beta2, epislon)
            self.logit_score_recorder = Sync_Single_Statistic_Recorder(channel, beta1, beta2, epislon)
        else:
            self.logit_recorder = Single_Statistic_Recorder(channel, beta1, beta2, epislon)
            self.score_recorder = Single_Statistic_Recorder(channel, beta1, beta2, epislon)
            self.logit_score_recorder = Single_Statistic_Recorder(channel, beta1, beta2, epislon)
        

    def forward(self, logit):
        x = logit
        with torch.no_grad():
            score = F.softmax(logit, -1)
            self.logit_recorder(logit)
            self.score_recorder(score)
            self.logit_score_recorder(torch.log(score))
        
        return x
    
    def clear(self):
        self.logit_recorder.clear()
        self.score_recorder.clear()
        self.logit_score_recorder.clear()