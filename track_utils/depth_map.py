import cv2
import os

from externals.PatchFusion.zoedepth.utils.config import get_config_user
from externals.PatchFusion.zoedepth.models.builder import build_model
import externals.PatchFusion.infer_user as PF
import numpy as np
from externals.PatchFusion.zoedepth.models.base_models.midas import Resize
from torchvision.transforms import Compose
import torch
import copy
from tqdm import tqdm
import torch.nn.functional as F


def detect_edges_binary_mask(binary_mask, pool_size, dilation_kernel_size):
    # Read the binary mask image
    binary_mask = binary_mask.astype(np.uint8)
    # Apply average pooling using cv2.blur
    pooled_image = cv2.blur(binary_mask.astype(float), (pool_size, pool_size))


    # Convert the result back to uint8
    pooled_image = pooled_image.astype(np.uint8)
    

    # Create an edge map by subtracting the pooled image from the original binary mask
    edge_map = cv2.absdiff(binary_mask, pooled_image)

    # Threshold the edge map
    _, edge_map = cv2.threshold(edge_map, 50, 255, cv2.THRESH_BINARY)

    # Apply dilation to thicken the edges
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    edge_map = cv2.dilate(edge_map, kernel, iterations=1)

    return binary_mask, edge_map

def calc_depth_diff(depth_map_1, depth_map_2):
    # Convert depth maps to grayscale
    depth_map_1_gray = cv2.cvtColor(depth_map_1, cv2.COLOR_BGR2GRAY)
    depth_map_2_gray = cv2.cvtColor(depth_map_2, cv2.COLOR_BGR2GRAY)

    # Calculate the absolute difference between the two depth maps
    diff = np.abs(depth_map_1_gray.astype(float) - depth_map_2_gray.astype(float))

    # Normalize the absolute difference to the range [0, 1]
    diff_normalized = cv2.normalize(diff, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Create an image where the red channel represents the absolute difference
    result_image = np.zeros((depth_map_1_gray.shape[0], depth_map_1_gray.shape[1], 3), dtype=np.uint8)
    result_image[:,:,2] = (diff_normalized * 255).astype(np.uint8)  # Set red channel based on the normalized difference

    return result_image


class DepthMapper:

    """

    Create a simple end-to-end patchfusion depth mapper

    """

    def __init__(self, ckp_path, model, model_cfg_path, img_resolution, mode, boundary, blr_mask):

        self.ckp_path = ckp_path
        self.model = model
        self.model_cfg_path = model_cfg_path
        self.img_resolution = img_resolution
        self.mode = mode
        self.boundary = boundary
        self.blr_mask = blr_mask

        self.crop_size = (int(img_resolution[0] // 4), int(img_resolution[1] // 4))

        self.overwrite_kwargs = {}
        self.overwrite_kwargs['model_cfg_path'] = model_cfg_path
        self.overwrite_kwargs["model"] = model
        self.transofrm = Compose([Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")])
        config = get_config_user(self.model, **self.overwrite_kwargs)
        config["pretrained_resource"] = ''
        self.model = build_model(config)
        self.model.cuda()
        self.model = PF.load_ckpt(self.model, self.ckp_path)
        self.model.eval()
        # self.model.cuda()
        self.model = self.model.to('cuda')


    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (W, H, C) (in BGR order).

        Returns:
            depth map: an image of shape (H, W, C)
        """

        img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB) / 255.0
        img = F.interpolate(torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2), self.img_resolution, mode='bicubic', align_corners=True)
        img = img.squeeze().permute(1, 2, 0)
        
        #(W,H, C)

        img = torch.tensor(img).unsqueeze(dim=0).permute(0, 3, 1, 2) # shape: 1, 3, w, h
        img_lr = self.transofrm(img)
        img = img.to('cuda')
        
        avg_depth_map = PF.regular_tile(self.model, img, offset_x=0, offset_y=0, img_lr=img_lr, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm )

        
        if self.mode== 'p16':
            pass
        elif self.mode== 'p49':
            PF.regular_tile(self.model, img, offset_x=self.crop_size[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)
            PF.regular_tile(self.model, img, offset_x=0, offset_y=self.crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)
            PF.regular_tile(self.model, img, offset_x=self.crop_size[1]//2, offset_y=self.crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)

        elif self.mode[0] == 'r':
            PF.regular_tile(self.model, img, offset_x=self.crop_size[1]//2, offset_y=0, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)
            PF.regular_tile(self.model, img, offset_x=0, offset_y=self.crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)
            PF.regular_tile(self.model, img, offset_x=self.crop_size[1]//2, offset_y=self.crop_size[0]//2, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)

            for i in tqdm(range(int(self.mode[1:]))):
                PF.random_tile(self.model, img, img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size = self.crop_size, img_resolution = self.img_resolution, transform=self.transofrm)

        
        color_depth_map = copy.deepcopy(avg_depth_map.average_map)
        color_depth_map = PF.colorize_infer(color_depth_map.detach().cpu().numpy())



        return color_depth_map, avg_depth_map.average_map


