# -*- coding: utf-8 -*-
# RUN IN PYTHON 3

import os
import sys
import csv
import cv2
import click
import random
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset

import torch
import torch.nn as nn
import torchvision


import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

model_save_path = "/home/venky/time_delay_vp/models/THDloss_model_04_11_2022_16_04/ACTVP_THD"

class Test:
    def __init__(self, model_save_path, model_save_name, model_name_save_appendix):
        self.list_of_p_measures = ["MAE", "MSE", "PSNR", "SSIM", "MAE_last", "MSE_last", "PSNR_last", "SSIM_last"]

        self.model = torch.load(model_save_path)
        self.features = model['features']

        self.test_features = features
        self.model.load_model(full_model=saved_model)
        saved_model = None
    
    def test_model(self):
        batch_losses = []
        self.model.set_test()
        for index, batch_features in enumerate(self.test_full_loader):
            print(str(index) + "\r")
            if batch_features[1].shape[0] == self.features["batch_size"]:                                               # messes up the models when running with a batch that is not complete
                groundtruth_scene, predictions_scene, groundtruth_tactile, prediction_tactile = self.format_and_run_batch(batch_features, test=True)              # run the model
                if self.test_features["quant_analysis"] == True and prediction_tactile == 100:
                    batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[self.features["n_past"]:], prediction_tactile))
                elif self.test_features["quant_analysis"] == True:
                    batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[self.features["n_past"]:], prediction_tactile, groundtruth_tactile[self.features["n_past"]]))

                if self.test_features["qual_analysis"] == True and index == 1:
                    self.save_images(predictions_scene, groundtruth_scene[self.features["n_past"]:], index)

        if self.test_features["quant_analysis"] == True:
            batch_losses = np.array(batch_losses)

            full_losses = [sum(batch_losses[:,0,i]) / batch_losses.shape[0] for i in range(batch_losses.shape[2])]
            last_ts_losses = [sum(batch_losses[:,1,i]) / batch_losses.shape[0] for i in range(batch_losses.shape[2])]

            full_losses = [float(i) for i in full_losses]
            last_ts_losses = [float(i) for i in last_ts_losses]

            if self.test_features["seen"]: data_save_path_append = "seen_"
            else:                          data_save_path_append = "unseen_"

            np.save(self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores.npy", batch_losses)
            lines = full_losses + last_ts_losses
            with open (self.test_features["data_save_path"] + data_save_path_append  + "test_loss_scores.txt", 'w') as f:
                for index, line in enumerate(lines):
                    f.write(self.list_of_p_measures[index] + ": " + str(line))
                    f.write('\n')


    features = {"model_name":model_name, "model_stage":model_stage, "model_folder_name":model_folder_name,
                "quant_analysis":quant_analysis, "qual_analysis":qual_analysis, "model_save_name":model_save_name,
                "qual_tactile_analysis":qual_tactile_analysis, "test_sample_time_step":test_sample_time_step,
                "model_name_save_appendix":model_name_save_appendix, "test_data_dir":test_data_dir, "scaler_dir":scaler_dir,
                "using_tactile_images":using_tactile_images, "using_depth_data":using_depth_data, "model_save_path":model_save_path,
                "data_save_path": data_save_path, "seen": seen, "device": device, "tactile_random": tactile_random, 
                "image_random": image_random}

if __name__ == "__main__":
    t = Test()
    t.test_model()