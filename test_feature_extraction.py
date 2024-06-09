import torch
from torchvision import models, transforms
import pandas as pd
import numpy as np
from out_of_domain.feature_extraction import save_gallery_features

t_image_folder = "test_images"
model_name = "resnet50"
output_file = f"{model_name}.npy"
save_gallery_features(t_image_folder, model_name)
print("Test passed: save_gallery_features function works as expected.")
    



