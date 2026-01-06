import os
import sys

sys.path.append('../')
print(os.getcwd())
from FaceModel import FaceModel

import torch
from backbones.vit_qs import VisionTransformer

import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

class QualityModel(FaceModel):
    def __init__(self, model_prefix, model_epoch, gpu_id,backbone, early_exit_block=12):
        self.early_exit_block = early_exit_block
        super(QualityModel, self).__init__(model_prefix, model_epoch, gpu_id,backbone)

    def _get_model(self, ctx, image_size, prefix, epoch, layer, backbone):
        if (backbone=="token"):
            backbones = VisionTransformer(
                img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=12,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode="token", early_exit_block=self.early_exit_block).to(f"cuda:{ctx}")

            dict_checkpoint = torch.load(os.path.join(prefix,"vitfiqa_token.pt"))
        elif (backbone=="crfiqa"):
            backbones = VisionTransformer(
                img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=12,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode="crfiqa", early_exit_block=self.early_exit_block).to(f"cuda:{ctx}")

            dict_checkpoint = torch.load(os.path.join(prefix,"vitfiqa_crfiqa.pt"))
        elif (backbone=="full"):
            backbones = VisionTransformer(
                img_size=112, patch_size=9, num_classes=512, embed_dim=512, depth=12,
                num_heads=8, drop_path_rate=0.1, norm_layer="ln", mask_ratio=0.1, mode="full", early_exit_block=self.early_exit_block).to(f"cuda:{ctx}")

            dict_checkpoint = torch.load(os.path.join(prefix,"vitfiqa_full.pt"))

        backbones.load_state_dict(dict_checkpoint)
        model = torch.nn.DataParallel(backbones, device_ids=[ctx])
        model.eval()
        return model

    @torch.no_grad()
    def _getFeatureBlob(self,input_blob):
        imgs = torch.Tensor(input_blob).cuda()
        imgs.div_(255).sub_(0.5).div_(0.5)
        feat, qs = self.model(imgs)
        return feat.cpu().numpy(), qs.cpu().numpy()
