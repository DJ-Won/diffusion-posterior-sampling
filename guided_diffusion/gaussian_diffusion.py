import math
import os
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch.optim as optim
from copy import deepcopy
import sys
sys.path.append("/home/ubuntu/dingjuwang/min/dps")
from util.img_utils import clear_color
from guided_diffusion.posterior_mean_variance import get_mean_processor, get_var_processor
from PIL import Image
import datetime
from copy import deepcopy



NUM_CLASSES = 1024 # from diffusion

__SAMPLER__ = {}

def index_to_log_onehot(x, num_classes):
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def log_onehot_to_index(log_x):
    return log_x.argmax(1)

def register_sampler(name: str):
    def wrapper(cls):
        if __SAMPLER__.get(name, None):
            raise NameError(f"Name {name} is already registered!") 
        __SAMPLER__[name] = cls
        return cls
    return wrapper


def get_sampler(name: str):
    if __SAMPLER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __SAMPLER__[name]


def create_sampler(sampler,
                   steps,
                   noise_schedule,
                   model_mean_type,
                   model_var_type,
                   dynamic_threshold,
                   clip_denoised,
                   rescale_timesteps,
                   timestep_respacing=""):
    
    sampler = get_sampler(name=sampler)
    
    betas = get_named_beta_schedule(noise_schedule, steps)
    if not timestep_respacing:
        timestep_respacing = [steps]
         
    return sampler(use_timesteps=space_timesteps(steps, timestep_respacing),
                   betas=betas,
                   model_mean_type=model_mean_type,
                   model_var_type=model_var_type,
                   dynamic_threshold=dynamic_threshold,
                   clip_denoised=clip_denoised, 
                   rescale_timesteps=rescale_timesteps)


class GaussianDiffusion:
    def __init__(self,
                 betas,
                 model_mean_type,
                 model_var_type,
                 dynamic_threshold,
                 clip_denoised,
                 rescale_timesteps,
                 temperature=1.0
                 ):

        # use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert self.betas.ndim == 1, "betas must be 1-D"
        assert (0 < self.betas).all() and (self.betas <=1).all(), "betas must be in (0..1]"

        self.num_timesteps = int(self.betas.shape[0])
        self.rescale_timesteps = rescale_timesteps

        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

        self.mean_processor = get_mean_processor(model_mean_type,
                                                 betas=betas,
                                                 dynamic_threshold=dynamic_threshold,
                                                 clip_denoised=clip_denoised)    
    
        self.var_processor = get_var_processor(model_var_type,
                                               betas=betas)

    def q_sample(self, x_start, t):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = extract_and_expand(self.sqrt_alphas_cumprod, t, x_start)
        coef2 = extract_and_expand(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return coef1 * x_start + coef2 * noise
    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root,
                      ):
        """
        The function used for sampling from noise.
        """ 
        img = x_start
        device = x_start.device

        pbar = tqdm(list(range(self.num_timesteps))[::-1])
        for idx in pbar:
            time = torch.tensor([idx] * img.shape[0], device=device)
            img = img.requires_grad_()
            out = self.p_sample(x=img, t=time, model=model)
            
            # Give condition.
            noisy_measurement = self.q_sample(measurement, t=time)

            # TODO: how can we handle argument for different condition method?
            img, distance = measurement_cond_fn(x_t=out['sample'],
                                      measurement=measurement,
                                      noisy_measurement=noisy_measurement,
                                      x_prev=img,
                                      x_0_hat=out['pred_xstart'])
            img = img.detach_()
           
            pbar.set_postfix({'distance': distance.item()}, refresh=False)
            if record:
                if idx % 100 == 0:
                    file_path = os.path.join(save_root, f"progress/x_{str(idx).zfill(4)}.png")
                    plt.imsave(file_path, clear_color(img))

        return img  
    
    def p_sample(self, model, x, t):
        raise NotImplementedError

    def p_mean_variance(self, model, x, t):
        model_output = model(x, self._scale_timesteps(t))
        
        # In the case of "learned" variance, model will give twice channels.
        if model_output.shape[1] == 2 * x.shape[1]:
            model_output, model_var_values = torch.split(model_output, x.shape[1], dim=1)
        else:
            # The name of variable is wrong. 
            # This will just provide shape information, and 
            # will not be used for calculating something important in variance.
            model_var_values = model_output

        model_mean, pred_xstart = self.mean_processor.get_mean_and_xstart(x, t, model_output)
        model_variance, model_log_variance = self.var_processor.get_variance(model_var_values, t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape

        return {'mean': model_mean,
                'variance': model_variance,
                'log_variance': model_log_variance,
                'pred_xstart': pred_xstart}

    
    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    elif isinstance(section_counts, int):
        section_counts = [section_counts]
    
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process. It can speed up the sampling speed
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = torch.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)


@register_sampler(name='ddpm')
class DDPM(SpacedDiffusion):
    def p_sample(self, model, x, t):
        out = self.p_mean_variance(model, x, t)
        sample = out['mean']

        noise = torch.randn_like(x)
        if t != 0:  # no noise when t == 0
            sample += torch.exp(0.5 * out['log_variance']) * noise

        return {'sample': sample, 'pred_xstart': out['pred_xstart']}

@register_sampler(name='g2d2')
class G2D2(SpacedDiffusion):
    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root,
                      text = 'a high-quality headshot of a girl',
                      num_class=4096,  #2887 for coco, 4096 for ithq
                      forget = 0.3,
                      operator=None,
                      content_seq_len=1024,
                      lambda_lr = 0.05,
                      base_lr = 0.5,
                      lambda_kl = 2.,
                      base_kl = 0.03,
                      sigma = 0.05,
                      dist_from_label = True,
                      label = None):

        """
        The function used for sampling from noise.
        forget: forget coefficient
        DL_balance: the balance between KL divergence and likelihood.
        """ 
        img = x_start
        device = x_start.device 
        measurement = measurement
        pbar = tqdm(list(range(self.num_timesteps))[::-1]) ## self.num_timesteps
        save_path_ = os.path.join(save_root,'auto','base_kl'+str(base_kl),'forget'+str(forget),'sigma'+str(sigma),'base_lr'+str(base_lr))
        os.makedirs(save_path_,exist_ok=True)
        
        p_alpha = None
        x0_pred_logit = None
        batch_size = 1
        zero_logits = torch.zeros((batch_size, num_class-1, content_seq_len),device=device) # num classes is from transformer, numclasses include
        one_logits = torch.ones((batch_size, 1, content_seq_len),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        xt_1_pred_logits = torch.log(mask_logits) # -inf occurs,    
        batch = {}
        batch['text'] = text
        condition = model.prepare_condition(batch=batch)
        cond_emb = model.transformer.condition_emb(condition['condition_token'])
        for idx in pbar:
            ###
            
            #mask

            t = torch.full((batch_size,), idx, device=device, dtype=torch.long)
            x0_pred_logit = model.transformer.predict_start(xt_1_pred_logits,cond_emb.type_as(xt_1_pred_logits),t).float()
            vq_theta = x0_pred_logit

            if p_alpha == None:
                p_alpha = (vq_theta).clone().detach().requires_grad_(True) 
            else:
                with torch.no_grad():
                    new_p_alpha = forget*p_alpha + (1-forget)*vq_theta
                    p_alpha.data.copy_(new_p_alpha)

            max_iter = 50

            # optimize alpha using adam(according to implementation) 优化的是logit
            LRate = base_lr*10**(lambda_lr/2*(2*idx/self.num_timesteps-1))
            DL_balance  = base_kl*10**(lambda_kl/2*(2*(idx)/self.num_timesteps-1))
            optimizer = optim.Adam([p_alpha], lr=LRate)
            for i in range(max_iter):
                sample = F.gumbel_softmax(p_alpha,dim=1,tau=1.)
                optimizer.zero_grad(set_to_none=True)
                p_alpha_ = F.softmax(p_alpha,dim=1)
                # calculate KL divergence
                # divergence = (sample*(sample.log()-vq_theta)).sum()
                divergence = (p_alpha_*((p_alpha_/torch.softmax(vq_theta,dim=1)).log())).sum()
                # calculate distance
                sample_img = model.content_codec.decode(sample,gumbel=True)/255*2-1
                degraded_img = operator.forward(sample_img)
                nll = ((measurement-degraded_img)**2).sum()/(2*sigma**2) # 
                # nll = ((measurement-degraded_img)**2).sqrt().sum()/(2*0.05**2)
                obj =   (1-DL_balance) * nll + DL_balance * divergence
                # obj = divergence
                obj.backward()
                optimizer.step()
                

                if i%10==0:
                    # print('\n')
                    # print('nll',nll)
                    # print('divergence',divergence)
                    # print('obj',obj)
                    pbar.set_postfix({'loss': obj}, refresh=False)
                # Give condition.
            
            x0_pred_logit = log_onehot_to_index(p_alpha)
            pbar.set_postfix({'loss': obj}, refresh=False)
            content_ = model.content_codec.decode(x0_pred_logit)
            # q(xt-1|x0)
            xt_1_pred_logits = model.transformer.q_sample(p_alpha,t-1)
            if record:
                if idx % 1 == 0:
                    content__ = content_.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                    for b in range(content__.shape[0]):
                        cnt = b
                        save_path = os.path.join(save_path_, f"x_{str(idx).zfill(4)}.png") 
                        im = Image.fromarray(content__[b])
                        im.save(save_path)
        label_recon_dist = None
        if dist_from_label:
            try:
                label_recon_dist = ((label-content_[0])**2).sum()
            except:
                raise NotImplementedError("label")
                
        return img,label_recon_dist


@register_sampler(name='ddim')
class DDIM(SpacedDiffusion):
    def p_sample(self, model, x, t, eta=0.0):
        out = self.p_mean_variance(model, x, t)
        
        eps = self.predict_eps_from_x_start(x, t, out['pred_xstart'])
        
        alpha_bar = extract_and_expand(self.alphas_cumprod, t, x)
        alpha_bar_prev = extract_and_expand(self.alphas_cumprod_prev, t, x)
        sigma = (
            eta
            * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = torch.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * torch.sqrt(alpha_bar_prev)
            + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        sample = mean_pred
        if t != 0:
            sample += sigma * noise
        
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def predict_eps_from_x_start(self, x_t, t, pred_xstart):
        coef1 = extract_and_expand(self.sqrt_recip_alphas_cumprod, t, x_t)
        coef2 = extract_and_expand(self.sqrt_recipm1_alphas_cumprod, t, x_t)
        return (coef1 * x_t - pred_xstart) / coef2

@register_sampler(name='ss')
class SS(SpacedDiffusion):
    def p_sample_loop(self,
                      model,
                      x_start,
                      measurement,
                      measurement_cond_fn,
                      record,
                      save_root,
                      text = 'a high-quality headshot of a girl',
                      num_class=4096,  #2887 for coco, 4096 for ithq
                      forget = 0.3,
                      operator=None,
                      content_seq_len=1024,
                      lambda_lr = 0.05,
                      base_lr = 0.5,
                      lambda_kl = 2.,
                      base_kl = 0.03):

        """
        The function used for sampling from noise.
        forget: forget coefficient
        DL_balance: the balance between KL divergence and likelihood.
        """ 
        img = x_start
        device = x_start.device 
        measurement = measurement
        pbar = tqdm(list(range(self.num_timesteps))[::-1]) ## self.num_timesteps

        time_now = datetime.datetime.now()
        time_now = str(time_now).replace(' ','_')
        os.makedirs(os.path.join(save_root,'ss'),exist_ok=True)
        
        p_alpha = None
        x0_pred_logit = None
        batch_size = 1
        zero_logits = torch.zeros((batch_size, num_class-1, content_seq_len),device=device) # num classes is from transformer, numclasses include
        one_logits = torch.ones((batch_size, 1, content_seq_len),device=device)
        mask_logits = torch.cat((zero_logits, one_logits), dim=1)
        xt_1_pred_logits = torch.log(mask_logits) # -inf occurs,    
        batch = {}
        batch['text'] = text
        condition = model.prepare_condition(batch=batch)
        cond_emb = model.transformer.condition_emb(condition['condition_token'])
        for idx in pbar:
            ###
            
            #mask

            t = torch.full((batch_size,), idx, device=device, dtype=torch.long)
            x0_pred_logit = model.transformer.predict_start(xt_1_pred_logits,cond_emb.type_as(xt_1_pred_logits),t).float()
            
            x0_pred_idx = log_onehot_to_index(x0_pred_logit)
            content_ = model.content_codec.decode(x0_pred_idx)
            # q(xt-1|x0)
            xt_1_pred_logits = model.transformer.q_sample(x0_pred_logit,t-1)
            if record:
                if idx % 1 == 0:
                    content_ = content_.permute(0, 2, 3, 1).to('cpu').numpy().astype(np.uint8)
                    for b in range(content_.shape[0]):
                        cnt = b
                        save_path = os.path.join(save_root,'base_kl_noinverse'+str(base_kl),'forget'+str(forget), f"x_{str(idx).zfill(4)}.png") 
                        im = Image.fromarray(content_[b])
                        im.save(save_path)
        return img
    

# =================
# Helper functions
# =================

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

# ================
# Helper function
# ================

def extract_and_expand(array, time, target):
    array = torch.from_numpy(array).to(target.device)[time].float()
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)
    return array.expand_as(target)


def expand_as(array, target):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array)
    elif isinstance(array, np.float):
        array = torch.tensor([array])
   
    while array.ndim < target.ndim:
        array = array.unsqueeze(-1)

    return array.expand_as(target).to(target.device)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
