from image_synthesis.utils.misc import instantiate_from_config


def create_model(config, args=None):
    return instantiate_from_config(config['model'])
