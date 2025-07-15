from model_factory import get_model

SUPPORTED_FEATURE_EXTRACTORS = {
    'osnet_x1_0',
    'resnet50',
    'mobilenetv2',
    'shufflenet',
    'pcb',
    'hacnn',
}

def model_supported(model_name):
    if model_name.lower() not in SUPPORTED_FEATURE_EXTRACTORS:
        raise ValueError(f"Model '{model_name}' is not supported")
    return model_name