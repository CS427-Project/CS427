"""
This module contains the function detect_face which takes in the image, model configuration, and model path, and outputs a saved image with the resulting bounding boxes and face landmarks
"""
from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

# ---------------------------
# Check if the GPU is available
# ---------------------------
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Helper functions for the detect_face method (no need to care)
# ---------------------------
def check_keys(model, pretrained_state_dict):
    """
    The purpose of this method is to check if the model weights being loaded are copmatible with 
    the model it is being loaded into
    """
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters sharing common prefix 'module.'
    The function remove_prefix is used to strip off a common prefix (e.g., 'module.') 
    from all keys (parameter names) in a PyTorch model's state_dict. This is just a step to fix naming problems when
    trying to load weights into the model
    """
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    """
    This method uses the check_keys() method and remove_prefix() methods to handle loading pretrained weights
    into the model
    """
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def return_model_cfg(cfg):
    if (cfg == "cfg_re50"):
        return cfg_re50
    elif (cfg == "cfg_mnet"):
        return cfg_mnet

# ---------------------------
# This is the main function in this .py file. It takes in the image path, model configuration, and model path, and outputs a saved image with the resulting bounding boxes and face landmarks
# ---------------------------
def detect_face(img_path, model_weights_path, output_img_path, test_config_json):
    cfg = return_model_cfg(test_config_json["cfg"])
    # Instantiate model
    net = RetinaFace(cfg=cfg, phase = 'test').to(device)
    net = load_model(net, model_weights_path, False)
    net.eval()
    # image_path = "./curve/test.jpg"
    img_raw = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    
    im_height, im_width, _ = img.shape
    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)
    
    # Miscellaneous params
    resize = test_config_json["resize"]
    confidence_threshold = test_config_json["confidence_threshold"]
    top_k = test_config_json["top_k"]
    nms_threshold = test_config_json["nms_threshold"]
    save_image = test_config_json["save_image"]
    vis_thres = test_config_json["vis_thres"]
    keep_top_k = test_config_json["keep_top_k"]
    
    tic = time.time()
    loc, conf, landms = net(img)  # forward pass
    print('net forward time: {:.4f}'.format(time.time() - tic))
    
    priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg_re50['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg_re50['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()
    
    # ignore low scores
    inds = np.where(scores > confidence_threshold)[0]
    boxes = boxes[inds]
    landms = landms[inds]
    scores = scores[inds]
    
    # keep top-K before NMS
    order = scores.argsort()[::-1][:top_k]
    boxes = boxes[order]
    landms = landms[order]
    scores = scores[order]
    
    # do NMS
    dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    keep = py_cpu_nms(dets, nms_threshold)
    # keep = nms(dets, nms_threshold,force_cpu=args.cpu)
    dets = dets[keep, :]
    landms = landms[keep]
    
    # keep top-K faster NMS
    dets = dets[:keep_top_k, :]
    landms = landms[:keep_top_k, :]
    
    dets = np.concatenate((dets, landms), axis=1)
    
    # show image
    if save_image:
        for b in dets:
            if b[4] < vis_thres:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
    
            # landms
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)
        # save image
        cv2.imwrite(output_img_path, img_raw)