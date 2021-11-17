from .DFTR import DFTR


def build_model(modelname, config):
    return {
            'dftr': DFTR,
           }[modelname.lower()](**config)
