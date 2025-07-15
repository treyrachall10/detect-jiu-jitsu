from torchreid.utils import FeatureExtractor

def get_model(model_name):
    if 'osnet' in model_name:
        return FeatureExtractor(model_name='osnet_x1_0', model_path='a/b/c/model.pth.tar', device='cuda')