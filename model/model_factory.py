from torchreid.utils import FeatureExtractor
from transformers import ViTModel, ViTImageProcessor
from model.backbones.vit_pytorch import vit_base_patch16_384_TransReID, deit_small_patch16_224_TransReID, vit_base_patch16_224_TransReID
import consts

def get_model(model_name):
    if 'osnet' in model_name.lower():
        return FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

    elif 'hugging-face' in model_name.lower():
        consts.processor = ViTImageProcessor.from_pretrained('facebook/dino-vits8')
        return ViTModel.from_pretrained("facebook/dino-vits8")

    elif 'transreid' in model_name.lower():
        model = deit_small_patch16_224_TransReID(
            img_size=(224, 224),
            stride_size=16,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            camera=0,
            view=0,
            local_feature=False,
            sie_xishu=1.0
        )
        weight_path = r"C:\Users\VrTeleop-01\Documents\bjj_project\reid_weights\vit_transreid_occ_duke.pth"
        model.load_param(weight_path)
        model.eval()
        return model

    else:
        raise ValueError(f"Model '{model_name}' not supported.")
