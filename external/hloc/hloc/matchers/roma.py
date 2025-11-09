import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / '../../third_party'))
from functools import partial
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
from typing import Union
from ..utils.base_model import BaseModel
from third_party.RoMa.romatch.utils.kde import kde
from third_party.RoMa.romatch.models.model_zoo.roma_models import roma_model, tiny_roma_v1_model

weight_urls = {
    "romatch": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
        "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
    },
    "tiny_roma_v1": {
        "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/tiny_roma_v1_outdoor.pth",
    },
    "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # hopefully this doesnt change :D
}


def tiny_roma_v1_outdoor(device, weights=None, xfeat=None):
    if weights is None:
        weights = torch.hub.load_state_dict_from_url(
            weight_urls["tiny_roma_v1"]["outdoor"], map_location=device
        )
    if xfeat is None:
        xfeat = torch.hub.load(
            "verlab/accelerated_features", "XFeat", pretrained=True, top_k=4096
        )

    return tiny_roma_v1_model(weights=weights, xfeat=xfeat).to(device)

class RoMa_Matches(BaseModel):
    def _init(self, conf):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        if conf['weight_mode'] == "indoor":
            self.weights = torch.hub.load_state_dict_from_url(
                weight_urls["romatch"]["indoor"], map_location=self.device
            )
            self.dinov2_weights = torch.hub.load_state_dict_from_url(
                weight_urls["dinov2"], map_location=self.device
            )
        elif conf['weight_mode'] == "outdoor":
            self.weights = torch.hub.load_state_dict_from_url(
                weight_urls["romatch"]["outdoor"], map_location=self.device
            )
            self.dinov2_weights = torch.hub.load_state_dict_from_url(
                weight_urls["dinov2"], map_location=self.device
            )
        self.amp_dtype: torch.dtype = torch.float16
        self.symmetric = True
        self.use_custom_corr = True
        self.upsample_preds = True
        self.coarse_res: Union[int, tuple[int, int]] = 560
        self.upsample_res: Union[int, tuple[int, int]] = 864
        self.max_keypoints = conf['max_keypoints']
        self.sample_mode = "threshold_balanced"
        self.sample_thresh = 100
    
    def _forward(self, data):
        feat0, feat1 = data[0], data[3]
        img0, img1 = data[1], data[4]
        img0_origin_sz, img1_origin_sz = data[2], data[5]
        w0, h0 = img0.size
        w1, h1 = img1.size
        W0, H0 = img0_origin_sz
        W1, H1 = img1_origin_sz
        self.coarse_res = (h0, w0)
        self.upsample_res = (int(round(H0))//2, int(round(W0))//2)
        model = roma_model(
            resolution=self.coarse_res,
            upsample_preds=self.upsample_preds,
            weights=self.weights,
            dinov2_weights=self.dinov2_weights,
            device=self.device,
            amp_dtype=self.amp_dtype,
            symmetric=self.symmetric,
            use_custom_corr=self.use_custom_corr,
            upsample_res=self.upsample_res
        )
        matches, mconf = model.match_from_feature(
            featureA=feat0,
            featureB=feat1,
            im_A_input=img0,
            im_B_input=img1
        )
        mconf = mconf[:, 0]
        num_max_keypoints = self.max_keypoints
        
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            mconf = mconf.clone()
            mconf[mconf > upper_thresh] = 1
        matches, mconf = (
            matches.reshape(-1, 4),
            mconf.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(
            mconf,
            num_samples=min(expansion_factor * num_max_keypoints, len(mconf)),
            replacement=False,
        )
        good_matches, good_certainty = matches[good_samples], mconf[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty
        density = kde(good_matches, std=0.1)
        p = 1 / (density + 1)
        p[density < 10] = (
            1e-7  # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        )
        balanced_samples = torch.multinomial(
            p, num_samples=min(num_max_keypoints, len(good_certainty)), replacement=False
        )
        matches, mconf = good_matches[balanced_samples], good_certainty[balanced_samples]

        # convert to pixel coordinates
        kpts0, kpts1 = model.to_pixel_coordinates(
            matches, H0, W0, H1, W1
        ) 
        
        pred = {
            "keypoints0": kpts0,
            "keypoints1": kpts1,
            "mconf": mconf,
        }
        
        extract_q = {
            "keypoints": kpts0,
            "scores": mconf
        }
        extract_ref = {
            "keypoints": kpts1,
            "scores": mconf
        }
        return pred, extract_q, extract_ref
    
    def pad_to(self, f, Ht, Wt):
        H, W = f.shape[-2:]
        pad_h = Ht - H
        pad_w = Wt - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return F.pad(f, (pad_left, pad_right, pad_top, pad_bottom), value=0)
    
    def pad_feature_pair(self, fq, fs, div=14):
        Hq, Wq = fq.shape[-2:]
        Hs, Ws = fs.shape[-2:]

        # Lấy chiều lớn nhất
        Ht = max(Hq, Hs)
        Wt = max(Wq, Ws)

        # Làm tròn lên để chia hết cho div
        Ht = (Ht + div - 1) // div * div
        Wt = (Wt + div - 1) // div * div
        fq_padded = self.pad_to(fq, Ht, Wt)
        fs_padded = self.pad_to(fs, Ht, Wt)

        return fq_padded, fs_padded, (Ht, Wt)