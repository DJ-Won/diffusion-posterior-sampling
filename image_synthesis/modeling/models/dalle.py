# ------------------------------------------
# VQ-Diffusion
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Shuyang Gu
# ------------------------------------------

import torch
import math
from torch import nn
from image_synthesis.utils.misc import instantiate_from_config
import time
import numpy as np
from PIL import Image
import os

from torch.cuda.amp import autocast

class DALLE(nn.Module):
    def __init__(
        self,
        *,
        content_info={'key': 'image'},
        condition_info={'key': 'text'},
        learnable_cf=False,
        content_codec_config,
        condition_codec_config,
        diffusion_config
    ):
        super().__init__()
        self.content_info = content_info
        self.condition_info = condition_info
        self.guidance_scale = 1.0
        self.learnable_cf = learnable_cf
        self.content_codec = instantiate_from_config(content_codec_config)
        self.condition_codec = instantiate_from_config(condition_codec_config)
        self.transformer = instantiate_from_config(diffusion_config)
        self.truncation_forward = False

    def parameters(self, recurse=True, name=None):
        if name is None or name == 'none':
            return super().parameters(recurse=recurse)
        else:
            names = name.split('+')
            params = []
            for n in names:
                try: # the parameters() method is not overwritten for some classes
                    params += getattr(self, name).parameters(recurse=recurse, name=name)
                except:
                    params += getattr(self, name).parameters(recurse=recurse)
            return params

    @property
    def device(self):
        return self.transformer.device

    def get_ema_model(self):
        return self.transformer

    @torch.no_grad()
    def prepare_condition(self, batch, condition=None): # condition for text
        cond_key = self.condition_info['key']
        cond = batch[cond_key] if condition is None else condition
        if torch.is_tensor(cond):
            cond = cond.to(self.device)
        cond = self.condition_codec.get_tokens(cond) # grad unrequired
        cond_ = {}
        for k, v in cond.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cond_['condition_' + k] = v
        return cond_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_content(self, batch, with_mask=False): # content for image
        cont_key = self.content_info['key']
        cont = batch[cont_key]
        if torch.is_tensor(cont):
            cont = cont.to(self.device)
        if not with_mask:
            cont = self.content_codec.get_tokens(cont)
        else:
            mask = batch['mask'.format(cont_key)]
            cont = self.content_codec.get_tokens(cont, mask, enc_with_mask=False)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        return cont_

    @autocast(enabled=False)
    @torch.no_grad()
    def prepare_input(self, batch):
        input = self.prepare_condition(batch)
        input.update(self.prepare_content(batch))
        return input

    def p_sample_with_truncation(self, func, sample_type):
        truncation_rate = float(sample_type.replace('q', ''))
        def wrapper(*args, **kwards):
            out = func(*args, **kwards)
            import random
            if random.random() < truncation_rate:
                out = func(out, args[1], args[2], **kwards)
            return out
        return wrapper


    def predict_start_with_truncation(self, func, sample_type):
        if sample_type[-1] == 'p':
            truncation_k = int(sample_type[:-1].replace('top', ''))
            content_codec = self.content_codec
            save_path = self.this_save_path
            def wrapper(*args, **kwards):
                out = func(*args, **kwards)
                val, ind = out.topk(k = truncation_k, dim=1)
                probs = torch.full_like(out, -70)
                probs.scatter_(1, ind, val)
                return probs
            return wrapper
        elif sample_type[-1] == 'r':
            truncation_r = float(sample_type[:-1].replace('top', ''))
            def wrapper(*args, **kwards):
                out = func(*args, **kwards) # transformer
                # notice for different batches, out are same, we do it on out[0]
                temp, indices = torch.sort(out, 1, descending=True) # biggest values and where to fetch
                temp1 = torch.exp(temp)
                temp2 = temp1.cumsum(dim=1)
                temp3 = temp2 < truncation_r # whether unmasked one is confidence
                new_temp = torch.full_like(temp3[:,0:1,:], True) # all true
                temp6 = torch.cat((new_temp, temp3), dim=1)# stack temp3 over 
                temp3 = temp6[:,:-1,:] # get rid of the last in temp3, which represent possibility of [MASK]
                temp4 = temp3.gather(1, indices.argsort(1)) # 
                temp5 = temp4.float()*out+(1-temp4.float())*(-70)
                probs = temp5
                return probs
            return wrapper

        else:
            print("wrong sample type")

    @torch.no_grad()
    def generate_content(
        self,
        *,
        batch,
        condition=None,
        filter_ratio = 0.5,
        temperature = 1.0,
        content_ratio = 0.0,
        replicate=1,
        return_att_weight=False,
        sample_type="top0.85r",
        start_step=0,
        predx0 =None
    ):
        self.eval()
        if condition is None:
            condition = self.prepare_condition(batch=batch)
        else:
            condition = self.prepare_condition(batch=None, condition=condition)
        
        batch_size = len(batch['text']) * replicate

        if self.learnable_cf: # co-efficient?
            cf_cond_emb = self.transformer.empty_text_embed.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            batch['text'] = [''] * batch_size
            cf_condition = self.prepare_condition(batch=batch)
            cf_cond_emb = self.transformer.condition_emb(cf_condition['condition_token']).float()
        
        def cf_predict_start(log_x_t, cond_emb, t):
            log_x_recon = self.transformer.predict_start(log_x_t, cond_emb, t)[:, :-1]
            if abs(self.guidance_scale - 1) < 1e-3:
                return torch.cat((log_x_recon, self.transformer.zero_vector), dim=1)
            cf_log_x_recon = self.transformer.predict_start(log_x_t, cf_cond_emb.type_as(cond_emb), t)[:, :-1]
            log_new_x_recon = cf_log_x_recon + self.guidance_scale * (log_x_recon - cf_log_x_recon)
            log_new_x_recon -= torch.logsumexp(log_new_x_recon, dim=1, keepdim=True)
            log_new_x_recon = log_new_x_recon.clamp(-70, 0)
            log_pred = torch.cat((log_new_x_recon, self.transformer.zero_vector), dim=1)
            return log_pred

        if replicate != 1:
            for k in condition.keys():
                if condition[k] is not None:
                    condition[k] = torch.cat([condition[k] for _ in range(replicate)], dim=0)
        if predx0 is None:
            content_token = self.prepare_content(batch=batch)
        else:
            content_token = {'content_token': predx0}

        if len(sample_type.split(',')) > 1:
            if sample_type.split(',')[1][:1]=='q':
                self.transformer.p_sample = self.p_sample_with_truncation(self.transformer.p_sample, sample_type.split(',')[1])
        if sample_type.split(',')[0][:3] == "top" and self.truncation_forward == False:
            self.transformer.cf_predict_start = self.predict_start_with_truncation(cf_predict_start, sample_type.split(',')[0]) # truncation, 截断
            self.truncation_forward = True

        if len(sample_type.split(',')) == 2 and sample_type.split(',')[1][:4]=='time' and int(float(sample_type.split(',')[1][4:])) >= 2:
            trans_out = self.transformer.sample_fast(condition_token=condition['condition_token'],
                                                condition_mask=condition.get('condition_mask', None),
                                                condition_embed=condition.get('condition_embed_token', None),
                                                content_token=content_token,
                                                filter_ratio=filter_ratio,
                                                temperature=temperature,
                                                return_att_weight=return_att_weight,
                                                return_logits=False,
                                                print_log=False,
                                                sample_type=sample_type,
                                                skip_step=int(float(sample_type.split(',')[1][4:])-1))

        else:
            if 'time' in sample_type and float(sample_type.split(',')[1][4:]) < 1:
                self.transformer.prior_ps = int(1024 // self.transformer.num_timesteps * float(sample_type.split(',')[1][4:]))
                if self.transformer.prior_rule == 0:
                    self.transformer.prior_rule = 1
                self.transformer.update_n_sample()
            trans_out = self.transformer.sample(condition_token=condition['condition_token'],
                                            condition_mask=condition.get('condition_mask', None),
                                            condition_embed=condition.get('condition_embed_token', None),
                                            content_token=content_token.get('content_token', None),
                                            filter_ratio=filter_ratio,
                                            temperature=temperature,
                                            return_att_weight=return_att_weight,
                                            return_logits=False,
                                            print_log=False,
                                            sample_type=sample_type,
                                            start_step=start_step)



        self.train()
        out = {
            'content': trans_out['content_token'],
            'x_start': trans_out['x_start'],
            'x_start_log':  trans_out['log_z']
        }
        

        return out

    @torch.no_grad()
    def reconstruct(
        self,
        input
    ):
        if torch.is_tensor(input):
            input = input.to(self.device)
        cont = self.content_codec.get_tokens(input)
        cont_ = {}
        for k, v in cont.items():
            v = v.to(self.device) if torch.is_tensor(v) else v
            cont_['content_' + k] = v
        rec = self.content_codec.decode(cont_['content_token'])
        return rec

    @torch.no_grad()
    def sample(
        self,
        img,
        clip = None,
        temperature = 1.,
        return_rec = True,
        filter_ratio = [0, 0.5, 1.0],
        content_ratio = [1], # the ratio to keep the encoded content tokens
        return_att_weight=False,
        return_logits=False,
        sample_type="normal",
        **kwargs,
    ):
        self.eval()
        #  No condition, already get the image, set condition to None
        condition = None
        content = img

        content_samples = {'input_image': img}

        for fr in filter_ratio:
            for cr in content_ratio:
                num_content_tokens = int((content['content_token'].shape[1] * cr))
                if num_content_tokens < 0:
                    continue
                else:
                    content_token = content['content_token'][:, :num_content_tokens]
                if sample_type == 'debug':
                    trans_out = self.transformer.sample_debug(condition_token=condition.get('condition_token', None),
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                else:
                    trans_out = self.transformer.sample(condition_token=condition.get('condition_token', None),
                                                        condition_mask=condition.get('condition_mask', None),
                                                        condition_embed=condition.get('condition_embed_token', None),
                                                        content_token=content_token,
                                                        filter_ratio=fr,
                                                        temperature=temperature,
                                                        return_att_weight=return_att_weight,
                                                        return_logits=return_logits,
                                                        content_logits=content.get('content_logits', None),
                                                        sample_type=sample_type,
                                                        **kwargs)

                content_samples['cond1_cont{}_fr{}_image'.format(cr, fr)] = self.content_codec.decode(trans_out['content_token'])

                if return_att_weight:
                    content_samples['cond1_cont{}_fr{}_image_condition_attention'.format(cr, fr)] = trans_out['condition_attention'] # B x Lt x Ld
                    content_att = trans_out['content_attention']
                    shape = *content_att.shape[:-1], self.content.token_shape[0], self.content.token_shape[1]
                    content_samples['cond1_cont{}_fr{}_image_content_attention'.format(cr, fr)] = content_att.view(*shape) # B x Lt x Lt -> B x Lt x H x W
                if return_logits:
                    content_samples['logits'] = trans_out['logits']
        self.train() 
        output = {'condition': batch[self.condition_info['key']]}   
        output.update(content_samples)
        return output

    def forward(
        self,
        batch,
        name='none',
        **kwargs
    ):
        input = self.prepare_input(batch)
        output = self.transformer(input, **kwargs)
        return output
