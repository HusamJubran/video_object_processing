import os
from PIL import Image
import glob
from os import path as osp
import sys
import torch
import cv2
from detectron2.structures import Instances
from detectron2.structures.boxes import Boxes

from segment_anything import build_sam, SamPredictor
from externals.segmentation import load_model
from externals.GroundingDINO.groundingdino.datasets import transforms as T
from externals.GroundingDINO.groundingdino.models import build_model
from externals.GroundingDINO.groundingdino.util.slconfig import SLConfig
from externals.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap


def get_category(input_string, string_list):
    for substring in string_list:
        if substring.lower() in input_string.lower():
            return substring
    return None  # Return None if no match is found

def find_matching_position(input_string, strings_list):
    words = input_string.split()
    for substring in strings_list:
        if substring in words:
            return words.index(substring)

    return -1  # Return -1 if no match is found


class SAMPredictor:

    """

    Create a simple end-to-end Sam predictor 

    """

    def __init__(self, grounding_text_input,coco_metadata_thing_classes):

        config_file = '/viscam/projects/video_animals/husam/prev_repos/flow_and_crop_together/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
        grounded_checkpoint = '/viscam/projects/video_animals/husam/prev_repos/flow_and_crop_together/utils/GroundingDINO/weights/groundingdino_swint_ogc.pth'
        sam_checkpoint = '/viscam/projects/video_animals/husam/prev_repos/flow_and_crop_together/utils/SAM_checkpoints/sam_vit_h_4b8939.pth'
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.device = 'cuda'
        self.model = load_model(config_file, grounded_checkpoint, self.device)
        self.predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))

        self.grounding_text_input = grounding_text_input
        self.category = get_category(grounding_text_input, coco_metadata_thing_classes )
        self.coco_metadata_thing_classes = coco_metadata_thing_classes

        

        

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """

        image_pil = Image.fromarray(original_image).convert("RGB")

        image = self.preprocess_image(image_pil)


        boxes_filt = self.get_grounding_output(
            self.model, image, self.grounding_text_input, self.box_threshold, self.text_threshold, device=self.device 
        )



        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(image)
        W, H = image_pil.size[:2]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        if boxes_filt.shape[0] == 0:
            return None
        
        #image HWC format

        transformed_boxes = self.predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)
        

        # masks are in BxCxHxW format, where C is the number of masks, and (H, W) is the original image size. but since multiple output mask is false, C=1
        #An array of shape BxC containing the model'spredictions for the quality of each mask. Again C=1
        masks, masks_scores, _ = self.predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        masks_scores = masks_scores.squeeze(dim = 1)
        masks = masks.squeeze(dim = 1)


        number_of_pred = boxes_filt.shape[0]

        output = {"instances": Instances(image.shape[:2], pred_classes=[self.coco_metadata_thing_classes.index(self.category)]*number_of_pred, scores = masks_scores, pred_boxes = Boxes(boxes_filt), pred_masks = masks)}
        return output


    def preprocess_image(self, original_image):


        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        
        image, _ = transform(original_image, None)  # 3, h, w
        return image

    def get_grounding_output(self, model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
        
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        model = model.to(device)
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (number of queries --> that come from the encoder and cross modality mechanism, 256 --> number of tokens)
        boxes = outputs["pred_boxes"].cpu()[0]  # (number of queries, 4 --> box corners)
        logits.shape[0]
        
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]




        return boxes_filt