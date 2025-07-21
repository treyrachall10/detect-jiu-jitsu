from torchreid.utils import FeatureExtractor
from transformers import ViTModel, ViTImageProcessor
import consts

def get_model(model_name):
    if 'osnet' in model_name.lower():
        return FeatureExtractor(model_name='osnet_x1_0', model_path='a/b/c/model.pth.tar', device='cuda')
    elif 'transformer' in model_name.lower():
        consts.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
        return ViTModel.from_pretrained("facebook/dino-vits8")
    else:
        raise ValueError(f"Model '{model_name}' not supported.")