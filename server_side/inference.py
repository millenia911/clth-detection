# import torch
import detectron2 as detectron2
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.4
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.RETINANET.NUM_CLASSES = 30
cfg.MODEL.WEIGHTS = "cloth-cp/model_final.pth"

img_cls = ['coats', 'rings', 'bag', 'boots', 'scarves', 'socks and stockings', 
    'shirts', 'shirts hidden under jackets', 'purses', 'windbreakers', 
    'jackets', 'belts', 'necklaces', 'ties', 'skirts', 'sunglasses', 
    'gloves and mittens', 'backpacks', 'overalls', 'pants', 'shorts', 
    'earrings', 'bracelets', 'watches', 'underwear', 'hats', 'swimwear', 
    'undefined', 'dresses', 'shoes']
    
MetadataCatalog.get("dataset_val").set(thing_classes=img_cls) 

def get_outfit_pred(im):
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    pred_cls = outputs["instances"].pred_classes.numpy()
    boxes = outputs["instances"].pred_boxes.tensor.numpy()
    pred_dict=[]
    for i, item in enumerate(pred_cls):
        _ = {"class": img_cls[item], "box": boxes[i].astype(int).tolist()}
        pred_dict.append(_)
    
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("dataset_val"), scale=1)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img_output = out.get_image()[:, :, ::-1]

    return img_output, pred_dict
