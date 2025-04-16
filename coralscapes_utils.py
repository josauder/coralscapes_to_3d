from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_to_id = {'human': 7,
 'background': 13,
 'fish': 9,
 'sand': 5,
 'rubble': 18,
 'unknown hard substrate': 12,
 'algae covered substrate': 10,
 'dark': 14,
 'branching bleached': 19,
 'branching dead': 20,
 'branching alive': 22,
 'stylophora alive': 34,
 'pocillopora alive': 31,
 'acropora alive': 25,
 'table acropora alive': 28,
 'table acropora dead': 32,
 'millepora': 21,
 'turbinaria': 27,
 'other coral bleached': 4,
 'other coral dead': 3,
 'other coral alive': 6,
 'massive/meandering alive': 17,
 'massive/meandering dead': 23,
 'massive/meandering bleached': 16,
 'meandering alive': 36,
 'meandering dead': 37,
 'meandering bleached': 33,
 'transect line': 15,
 'transect tools': 8,
 'sea urchin': 35,
 'sea cucumber': 26,
 'anemone': 30,
 'sponge': 29,
 'clam': 24,
 'other animal': 11,
 'trash': 2,
 'seagrass': 1,
 'crown of thorn': 38,
 'dead clam': 39}


id_to_class = {v:k for k,v in class_to_id.items()}
colors = {"human": [255, 0, 0], "background": [29, 162, 216], "fish": [255, 255, 0], "sand": [194, 178, 128], "rubble": [161, 153, 128], "unknown hard substrate": [125, 125, 125], "algae covered substrate": [125, 163, 125], "dark": [31, 31, 31], "branching bleached": [252, 231, 240], "branching dead": [123, 50, 86], "branching alive": [226, 91, 157], "stylophora alive": [255, 111, 194], "pocillopora alive": [255, 146, 150], "acropora alive": [236, 128, 255], "table acropora alive": [189, 119, 255], "table acropora dead": [85, 53, 116], "millepora": [244, 150, 115], "turbinaria": [228, 255, 119], "other coral bleached": [250, 224, 225], "other coral dead": [114, 60, 61], "other coral alive": [224, 118, 119], "massive/meandering alive": [236, 150, 21], "massive/meandering dead": [134, 86, 18], "massive/meandering bleached": [255, 248, 228], "meandering alive": [230, 193, 0], "meandering dead": [119, 100, 14], "meandering bleached": [251, 243, 216], "transect line": [0, 255, 0], "transect tools": [8, 205, 12], "sea urchin": [0, 142, 255], "sea cucumber": [0, 231, 255], "anemone": [0, 255, 189], "sponge": [240, 80, 80], "clam": [189, 255, 234], "other animal": [0, 255, 255], "trash": [255, 0, 134], "seagrass": [125, 222, 125], "crown of thorn": [179, 245, 234], "dead clam": [89, 155, 134]}
colors = {class_to_id[k]:np.array(v).astype(np.uint8) for k, v in colors.items()}

preprocessor = SegformerImageProcessor.from_pretrained("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024")
model = SegformerForSemanticSegmentation.from_pretrained("EPFL-ECEO/segformer-b5-finetuned-coralscapes-1024-1024").to(device)



def resize_image(image, target_size=1024):
    """
    Used to resize the image such that the smaller side equals 1024
    """
    h_img, w_img = image.size
    if h_img < w_img:
        new_h, new_w = target_size, int(w_img * (target_size / h_img))
    else:
        new_h, new_w  = int(h_img * (target_size / w_img)), target_size
    resized_img = image.resize((new_h, new_w))
    return resized_img


def segment_image(image, preprocessor, model, crop_size = (1024, 1024), num_classes = 40, transform=None, do_not_predict_background=False):
    """
    Finds an optimal stride based on the image size and aspect ratio to create
    overlapping sliding windows of size 1024x1024 which are then fed into the model.  
    """ 
    h_crop, w_crop = crop_size
    
    img = torch.Tensor(np.array(resize_image(image, target_size=1024)).transpose(2, 0, 1)).unsqueeze(0)
    batch_size, _, h_img, w_img = img.size()
    
    if transform:
        img = torch.Tensor(transform(image = img.numpy())["image"]).to(device)    
        
    h_grids = int(np.round(3/2*h_img/h_crop)) if h_img > h_crop else 1
    w_grids = int(np.round(3/2*w_img/w_crop)) if w_img > w_crop else 1
    
    h_stride = int((h_img - h_crop + h_grids -1)/(h_grids -1)) if h_grids > 1 else h_crop
    w_stride = int((w_img - w_crop + w_grids -1)/(w_grids -1)) if w_grids > 1 else w_crop
    
    preds = img.new_zeros((batch_size, num_classes, h_img, w_img), device="cuda")
    count_mat = img.new_zeros((batch_size, 1, h_img, w_img), device="cuda")
    
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            with torch.no_grad():
                if(preprocessor):
                    inputs = preprocessor(crop_img, return_tensors = "pt")
                    inputs["pixel_values"] = inputs["pixel_values"].to(device)
                else:
                    inputs = crop_img.to(device)
                outputs = model(**inputs)

            resized_logits = F.interpolate(
                outputs.logits[0].unsqueeze(dim=0), size=crop_img.shape[-2:], mode="bilinear", align_corners=False
            )
            preds += F.pad(resized_logits,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0
    preds = preds / count_mat
    preds = F.interpolate(preds, size=image.size[::-1], mode='nearest')
    if do_not_predict_background:
        preds[:,class_to_id['background']] -= 20000
    preds = preds.argmax(dim=1)
    preds = preds.unsqueeze(0).type(torch.uint8)
    label_pred = preds.squeeze().cpu().numpy()
    return label_pred


def segment(image,do_not_predict_background=False):
    global model
    global preprocessor
    return segment_image(image, preprocessor, model, do_not_predict_background=do_not_predict_background)