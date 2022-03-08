# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms, datasets

from . import networks
from .utils import download_model_if_doesnt_exist

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def estimate_depthmap(img_batch):
    """Function to predict for a single image or folder of images
    """
    model_name = "mono_640x192"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    download_model_if_doesnt_exist(model_name)
    model_path = os.path.join("models", model_name)
    print("-> Loading model from ", model_path)
    
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()



    # PREDICTING ON EACH IMAGE IN TURN
    #out_arr = np.zeros((8, 1, 192, 640))
    #out_arr_c = np.zeros((8, 480, 640, 3))
    with torch.no_grad():
        #input_image = input_image.unsqueeze(0)

        # PREDICTION
        input_image = img_batch.to(device)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp_resized = outputs[("disp", 0)]

        # Saving colormapped depth image
        disp_resized_np = disp_resized.cpu().numpy()
        scaled_disp, depth = disp_to_depth(disp_resized_np, 0.1, 100)
        #return scaled_disp
        #out_arr[i] = scaled_disp
    return scaled_disp.squeeze()
        
"""            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            return colormapped_im
            #out_arr_c[i] = colormapped_im"""
        


    #, out_arr_c