from image_synthesis.utils.misc import instantiate_from_config
import os
import yaml
import torch

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_model(ema, model_path, config_path, imagenet_cf):
    if 'OUTPUT' in model_path: # pretrained model
        model_name = model_path.split(os.path.sep)[-3]
    else: 
        model_name = os.path.basename(config_path).replace('.yaml', '')

    config = load_yaml(config_path)

    if imagenet_cf:
        config['model']['params']['diffusion_config']['params']['transformer_config']['params']['class_number'] = 1001

    model = create_model(config)
    model_parameters = get_model_parameters_info(model)
    
    print(model_parameters)
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location="cpu")
    else:
        print("Model path: {} does not exist.".format(model_path))
        exit(0)
    if 'last_epoch' in ckpt:
        epoch = ckpt['last_epoch']
    elif 'epoch' in ckpt:
        epoch = ckpt['epoch']
    else:
        epoch = 0

    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    print('Model missing keys:\n', missing)
    print('Model unexpected keys:\n', unexpected)

    if ema==True and 'ema' in ckpt:
        print("Evaluate EMA model")
        ema_model = model.get_ema_model()
        missing, unexpected = ema_model.load_state_dict(ckpt['ema'], strict=False)
    
    return {'model': model, 'epoch': epoch, 'model_name': model_name, 'parameter': model_parameters}

def create_model(config, args=None):
    return instantiate_from_config(config['model'])

def get_model_parameters_info(model):
    # for mn, m in model.named_modules():
    parameters = {'overall': {'trainable': 0, 'non_trainable': 0, 'total': 0}}
    for child_name, child_module in model.named_children():
        parameters[child_name] = {'trainable': 0, 'non_trainable': 0}
        for pn, p in child_module.named_parameters():
            if p.requires_grad:
                parameters[child_name]['trainable'] += p.numel()
            else:
                parameters[child_name]['non_trainable'] += p.numel()
        parameters[child_name]['total'] = parameters[child_name]['trainable'] + parameters[child_name]['non_trainable']
        
        parameters['overall']['trainable'] += parameters[child_name]['trainable']
        parameters['overall']['non_trainable'] += parameters[child_name]['non_trainable']
        parameters['overall']['total'] += parameters[child_name]['total']
    
    # format the numbers
    def format_number(num):
        K = 2**10
        M = 2**20
        G = 2**30
        if num > G: # K
            uint = 'G'
            num = round(float(num)/G, 2)
        elif num > M:
            uint = 'M'
            num = round(float(num)/M, 2)
        elif num > K:
            uint = 'K'
            num = round(float(num)/K, 2)
        else:
            uint = ''
        
        return '{}{}'.format(num, uint)
    
    def format_dict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                format_dict(v)
            else:
                d[k] = format_number(v)
    
    format_dict(parameters)
    return parameters