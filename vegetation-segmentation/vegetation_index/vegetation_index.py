import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

# sum pixels or sum the pre-thresholded softmaxes?
# aggregate vegetation/grass/trees? Maybe just vegetation?
def vegetation_index(mask, depth):
    depth_nearby_mask = depth > 1/0.5
    depth_actual = 1 / depth
    depth_actual_masked = depth_actual * depth_nearby_mask
    
    mask_grass      = mask[:,1,:,:]
    mask_vegetation = mask[:,2,:,:]
    
    pixels_grass      = mask_grass * depth_actual_masked
    pixels_vegetation = mask_vegetation * depth_actual_masked
    
    count_grass = pixels_grass.sum(axis=1).sum(axis=1)
    count_vegetation = pixels_vegetation.sum(axis=1).sum(axis=1)
    
    return {"grass": count_grass, "vegetation": count_vegetation}

def score(x_true, x_pred):
    #scale_factor = x_pred[0]/x_true[0]
    #x_pred_scaled = x_pred / scale_factor
    #score = mean_absolute_percentage_error(x_true, x_pred_scaled)
    return np.corrcoef(x_true, x_pred)[0,1]#, x_pred_scaled