import torch

p_alpha_ = torch.rand([2,2])
vq_theta = torch.rand([])

divergence = torch.kl_div(torch.log(p_alpha_),vq_theta).sum()
divergence1 = (p_alpha_*torch.log(p_alpha_/(vq_theta))).sum()