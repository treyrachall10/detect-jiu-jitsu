SUPPORTED_FEATURE_EXTRACTORS = {
    'osnet_x1_0',
    'resnet50',
    'transformer',
    'transreid',
    'mobilenetv2',
    'shufflenet',
    'pcb',
    'hacnn',
}

def model_supported(model_name):
    if model_name.lower() not in SUPPORTED_FEATURE_EXTRACTORS:
        raise ValueError(f"Model '{model_name}' is not supported")
    return model_name
