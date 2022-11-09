# -*- coding: utf-8 -*-
# RUN IN PYTHON 3
import os
import csv
import cv2
import copy
import math
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import(AutoMinorLocator, MultipleLocator)
from matplotlib.animation import FuncAnimation

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm
from pickle import load
from datetime import datetime
from torch.utils.data import Dataset

from tdvp import ACTVP as ACTVP

model_path      = "/home/venky/time_delay_vp/models/THDloss_model_04_11_2022_16_04/ACTVP_THD"
data_save_path  = "/home/venky/time_delay_vp/models/THDloss_model_04_11_2022_16_04/scaled_test/"
test_data_dir   = "/home/venky/time_delay_vp/test_out/test_trial_0/"
scaler_dir      = "/home/venky/time_delay_vp/scalar_info_universal/"

seed = 42
epochs = 100
batch_size = 32
learning_rate = 1e-3
context_frames = 5
sequence_length = 15

train_percentage = 0.9
validation_percentage = 0.1

torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available


class BatchGenerator:
    def __init__(self):
        self.data_map = []
        with open(test_data_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map)
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        self.data_map = []
        return test_loader


class FullDataSet:
    def __init__(self, data_map):
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_task_data = np.load(test_data_dir + value[0])
        robot_joint_data = np.load(test_data_dir + value[1])

        side_camera_image_names = []
        for image_name in np.load(test_data_dir + value[3]):
            side_camera_image_names.append(np.load(test_data_dir + image_name))

        experiment_number = np.load(test_data_dir + value[4])
        time_steps = np.load(test_data_dir + value[5])
        return [robot_task_data.astype(np.float32), robot_joint_data.astype(np.float32), np.array(side_camera_image_names).astype(np.float32), np.array(experiment_number).astype(np.float32), np.array(time_steps).astype(np.float32)]


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return(torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device))

# class ACTVP(nn.Module):
#     def __init__(self):
#         super(ACTVP, self).__init__()
#         self.convlstm1 = ConvLSTMCell(input_dim=3, hidden_dim=12, kernel_size=(3, 3), bias=True).cuda()
#         self.convlstm2 = ConvLSTMCell(input_dim=24, hidden_dim=24, kernel_size=(3, 3), bias=True).cuda()
#         self.conv1 = nn.Conv2d(in_channels=27, out_channels=3, kernel_size=5, stride=1, padding=2).cuda()
#
#     def forward(self, tactiles, actions):
#         self.batch_size = actions.shape[1]
#         state = actions[0]
#         state.to(device)
#         batch_size__ = tactiles.shape[1]meta
#         hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
#         hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
#         outputs = []
#         for index,(sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
#             sample_tactile.to(device)
#             sample_action.to(device)
#             # 2. Run through lstm:
#             if index > context_frames-1:
#                 hidden_1, cell_1 = self.convlstm1(input_tensor=output, cur_state=[hidden_1, cell_1])
#                 state_action = torch.cat((state, sample_action), 1)
#                 robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
#                 hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
#                 skip_connection_added = torch.cat((hidden_2, output), 1)
#                 output = self.conv1(skip_connection_added)
#                 outputs.append(output)
#             else:
#                 hidden_1, cell_1 = self.convlstm1(input_tensor=sample_tactile, cur_state=[hidden_1, cell_1])
#                 state_action = torch.cat((state, sample_action), 1)
#                 robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
#                 hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
#                 skip_connection_added = torch.cat((hidden_2, sample_tactile), 1)
#                 output = self.conv1(skip_connection_added)
#                 last_output = output
#         outputs = [last_output] + outputs
#         return torch.stack(outputs)


class ACTVP_double(nn.Module):
    def __init__(self, device, context_frames):
        super(ACTVP_double, self).__init__()
        self.device = device
        self.context_frames = context_frames
        self.convlstm1 = ConvLSTMCell(input_dim=3, hidden_dim=12, kernel_size=(3, 3), bias=True, device=self.device).cuda()
        self.convlstm2 = ConvLSTMCell(input_dim=24, hidden_dim=24, kernel_size=(3, 3), bias=True, device=self.device).cuda()
        self.conv1 = nn.Conv2d(in_channels=27, out_channels=24, kernel_size=5, stride=1, padding=2).cuda()
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=3, kernel_size=5, stride=1, padding=2).cuda()

    def forward(self, tactiles, actions):
        self.batch_size = actions.shape[1]
        state = actions[0]
        state.to(self.device)
        batch_size__ = tactiles.shape[1]
        hidden_1, cell_1 = self.convlstm1.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
        hidden_2, cell_2 = self.convlstm2.init_hidden(batch_size=self.batch_size, image_size=(32, 32))
        outputs = []
        for index, (sample_tactile, sample_action) in enumerate(zip(tactiles[0:-1].squeeze(), actions[1:].squeeze())):
            sample_tactile.to(self.device)
            sample_action.to(self.device)
            # 2. Run through lstm:
            if index > self.context_frames-1:
                hidden_1, cell_1 = self.convlstm1(input_tensor=output, cur_state=[hidden_1, cell_1])
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
                skip_connection_added = torch.cat((hidden_2, output), 1)
                output1 = self.conv1(skip_connection_added)
                output = self.conv2(output1)
                outputs.append(output)
            else:
                hidden_1, cell_1 = self.convlstm1(input_tensor=sample_tactile, cur_state=[hidden_1, cell_1])
                state_action = torch.cat((state, sample_action), 1)
                robot_and_tactile = torch.cat((torch.cat(32*[torch.cat(32*[state_action.unsqueeze(2)], axis=2).unsqueeze(3)], axis=3), hidden_1.squeeze()), 1)
                hidden_2, cell_2 = self.convlstm2(input_tensor=robot_and_tactile, cur_state=[hidden_2, cell_2])
                skip_connection_added = torch.cat((hidden_2, sample_tactile), 1)
                output1 = self.conv1(skip_connection_added)
                output = self.conv2(output1)
                last_output = output
        outputs = [last_output] + outputs
        return torch.stack(outputs)


class image_player():
    def __init__(self, images, save_name, feature, experiment_to_test):
        self.feature = feature
        self.save_name = save_name
        self.experiment_to_test = experiment_to_test
        self.file_save_name = data_save_path + 'test_plots_' + str(self.experiment_to_test) + '/' + self.save_name + '_feature_' + str(self.feature) + '.gif'
        print(self.file_save_name)
        self.run_the_tape(images)

    def grab_frame(self):
        frame = self.images[self.indexyyy][:, :, self.feature] * 255
        return frame

    def update(self, i):
        plt.title(i)
        self.im1.set_data(self.grab_frame())
        self.indexyyy += 1
        if self.indexyyy == len(self.images):data_save_path
        ax1 = plt.subplot(1, 2, 1)
        self.im1 = ax1.imshow(self.grab_frame(), cmap='gray', vmin=0, vmax=255)
        ani = FuncAnimation(plt.gcf(), self.update, interval=20.8, save_count=len(images), repeat=False)
        ani.save(self.file_save_name)


class ModelTester:
    def __init__(self):
        self.test_full_loader = BG.load_full_data()


        # load model:
        # self.full_model = ACTVP_double(device, context_frames)
        self.full_model = torch.load(model_path)
        self.full_model.eval()

        self.criterion = nn.L1Loss()
        self.optimizer = optim.Adam(self.full_model.parameters(), lr=learning_rate)

        self.load_scalars()

    def test_full_model(self):
        self.performance_data = []
        self.prediction_data = []
        self.tg_back_scaled = []
        self.tp1_back_scaled = []
        self.tp5_back_scaled = []
        self.tp10_back_scaled = []
        self.current_exp = 0

        for index, batch_features in enumerate(self.test_full_loader):
            side_camera = batch_features[2].permute(1, 0, 4, 3, 2).to(device)
            robot_joint = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            side_camera_predictions = self.full_model.forward(side_camera=side_camera,
                                                           robot_joint_data=robot_joint)  # Step 3. Run our forward pass.

            experiment_number = batch_features[3][context_frames:]
            time_steps = batch_features[4].permute(1, 0)[context_frames:]
            # self.meta = batch_features[5][0]

            current_batch = 0
            new_batch = 0
            for index, exp in enumerate(experiment_number.T):
                if exp.item() == self.current_exp:
                    current_batch += 1
                else:
                    new_batch += 1

            for i in [0, 1]:
                if i == 0:
                    side_camera_cut = side_camera[:, 0:current_batch, :, :, :]
                    side_camera_predictions_cut = side_camera_predictions[:, 0:current_batch, :, :, :]
                    experiment_number_cut = experiment_number[0:current_batch]
                    time_steps_cut = time_steps[:, 0:current_batch]
                if i == 1:
                    side_camera_cut = side_camera[:, current_batch:, :, :, :]
                    side_camera_predictions_cut = side_camera_predictions[:, current_batch:, :, :, :]
                    experiment_number_cut = experiment_number[current_batch:]
                    time_steps_cut = time_steps[:, current_batch:]

                self.prediction_data.append(
                    [side_camera_predictions_cut.cpu().detach(), side_camera_cut[context_frames:].cpu().detach(),
                     experiment_number_cut.cpu().detach(), time_steps_cut.cpu().detach()])

                # convert back to 48 feature tactile readings for plotting:
                gt = []
                p1 = []
                p5 = []
                p10 = []
                for batch_value in range(side_camera_predictions_cut.shape[1]):
                    gt.append(side_camera_cut[context_frames - 1][batch_value].permute(1, 2, 0).cpu().detach().numpy())
                    p1.append(side_camera_predictions_cut[0][batch_value].permute(1, 2, 0).cpu().detach().numpy())
                    p5.append(side_camera_predictions_cut[4][batch_value].permute(1, 2, 0).cpu().detach().numpy())
                    p10.append(side_camera_predictions_cut[9][batch_value].permute(1, 2, 0).cpu().detach().numpy())

                gt = np.array(gt)
                p1 = np.array(p1)
                p5 = np.array(p5)
                p10 = np.array(p10)
                descalled_data = []

                # print(gt.shape, p1.shape, p5.shape, p10.shape, gt.shape[0])
                dims_gt = gt.shape
                dims_p1 = p1.shape
                dims_p5 = p5.shape
                dims_p10 = p10.shape
                if gt.shape[0] != 0:
                    for index, data in enumerate([gt, p1, p5, p10]):
                        side_cam_inverse_minmax = [self.side_cam_minmax_scaler_all[feature].inverse_transform(data[:,:,:,feature].reshape(-1,1)) for feature in range(3)]
                        side_cam_inv_scaler = [self.side_cam_std_scaler_all[feature].inverse_transform(side_cam_inverse_minmax[feature]) for feature in range(3)]
                        
                        
                        if index == 0:
                            side_cam_inv_scaler_r = side_cam_inv_scaler[0].reshape(dims_gt[0], dims_gt[1], dims_gt[2],1)
                            side_cam_inv_scaler_g = side_cam_inv_scaler[1].reshape(dims_gt[0], dims_gt[1], dims_gt[2],1)
                            side_cam_inv_scaler_b = side_cam_inv_scaler[2].reshape(dims_gt[0], dims_gt[1], dims_gt[2],1)
                            side_cam_inv_scaler = np.concatenate((side_cam_inv_scaler_r, side_cam_inv_scaler_g, side_cam_inv_scaler_b), axis=3)
                            descalled_data.append(side_cam_inv_scaler)
                            # self.tg_back_scaled.append(side_cam_inv_scaler)  # descalled_data[0])
                        elif index == 1:
                            side_cam_inv_scaler_r = side_cam_inv_scaler[0].reshape(dims_p1[0], dims_p1[1], dims_p1[2],1)
                            side_cam_inv_scaler_g = side_cam_inv_scaler[1].reshape(dims_p1[0], dims_p1[1], dims_p1[2],1)
                            side_cam_inv_scaler_b = side_cam_inv_scaler[2].reshape(dims_p1[0], dims_p1[1], dims_p1[2],1)
                            side_cam_inv_scaler = np.concatenate((side_cam_inv_scaler_r, side_cam_inv_scaler_g, side_cam_inv_scaler_b), axis=3)
                            descalled_data.append(side_cam_inv_scaler)
                            # self.tp1_back_scaled.append(side_cam_inv_scaler)  # descalled_data[1])
                        elif index == 2:
                            side_cam_inv_scaler_r = side_cam_inv_scaler[0].reshape(dims_p5[0], dims_p5[1], dims_p5[2],1)
                            side_cam_inv_scaler_g = side_cam_inv_scaler[1].reshape(dims_p5[0], dims_p5[1], dims_p5[2],1)
                            side_cam_inv_scaler_b = side_cam_inv_scaler[2].reshape(dims_p5[0], dims_p5[1], dims_p5[2],1)
                            side_cam_inv_scaler = np.concatenate((side_cam_inv_scaler_r, side_cam_inv_scaler_g, side_cam_inv_scaler_b), axis=3)
                            descalled_data.append(side_cam_inv_scaler)
                            # self.tp5_back_scaled.append(side_cam_inv_scaler)  # descalled_data[2])
                        elif index == 3:
                            side_cam_inv_scaler_r = side_cam_inv_scaler[0].reshape(dims_p10[0], dims_p10[1], dims_p10[2],1)
                            side_cam_inv_scaler_g = side_cam_inv_scaler[1].reshape(dims_p10[0], dims_p10[1], dims_p10[2],1)
                            side_cam_inv_scaler_b = side_cam_inv_scaler[2].reshape(dims_p10[0], dims_p10[1], dims_p10[2],1)
                            side_cam_inv_scaler = np.concatenate((side_cam_inv_scaler_r, side_cam_inv_scaler_g, side_cam_inv_scaler_b), axis=3)
                            descalled_data.append(side_cam_inv_scaler)
                            # self.tp10_back_scaled.append(side_cam_inv_scaler)  # descalled_data[3])
                    #     (r, g, b) = np.split(data, 3, axis=0)
                    #     side_cam_r_inverse_minmax = self.min_max_scalerr_full_data.inverse_transform(r.reshape(-1,1))
                    #     side_cam_g_inverse_minmax = self.min_max_scalerr_full_data.inverse_transform(g.reshape(-1,1))
                    #     side_cam_b_inverse_minmax = self.min_max_scalerr_full_data.inverse_transform(b.reshape(-1,1))
                    #     side_cam_r_inverse_full = self.scaler_side_cam_r.inverse_transform(side_cam_r_inverse_minmax)
                    #     side_cam_g_inverse_full = self.scaler_side_cam_g.inverse_transform(side_cam_g_inverse_minmax)
                    #     side_cam_b_inverse_full = self.scaler_side_cam_b.inverse_transform(side_cam_b_inverse_minmax)
                    #     descalled_data.append(
                    #         np.concatenate((side_cam_r_inverse_full, side_cam_g_inverse_full, side_cam_b_inverse_full),
                    #                         axis=1))

                    self.tg_back_scaled.append(gt)  # descalled_data[0])
                    self.tp1_back_scaled.append(p1)  # descalled_data[1])
                    self.tp5_back_scaled.append(p5)  # descalled_data[2])
                    self.tp10_back_scaled.append(p10)  # descalled_data[3])

                if i == 0 and new_batch != 0:
                    print ("currently testing trial number: ", str (self.current_exp))
                    # self.calc_train_trial_performance()
                    self.calc_trial_performance()
                    self.save_predictions(self.current_exp)
                    # self.create_test_plots(self.current_exp)
                    # self.create_difference_gifs(self.current_exp)
                    self.prediction_data = []
                    self.tg_back_scaled = []
                    self.tp1_back_scaled = []
                    self.tp5_back_scaled = []
                    self.tp10_back_scaled = []
                    self.current_exp += 1
                if i == 0 and new_batch == 0:
                    break

        print ("Hello :D ")

        self.calc_test_performance()
        # self.calc_train_performance()

    def calc_train_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        # meta_data_file_name = str(np.load(self.meta)[0]) + "/meta_data.csv"
        # meta_data = []
        # with open(meta_data_file_name, 'r') as f:  # rb
        #     reader = csv.reader(f)
        #     for row in reader:
        #         meta_data.append(row)
        # seen = meta_data[1][2]
        # object = meta_data[1][0]

        index = 0
        with torch.no_grad():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan(mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss    += mae_loss_check
                    mae_loss_1  += self.criterion(batch_set[0][0], batch_set[1][0]).item()
                    mae_loss_5  += self.criterion(batch_set[0][4], batch_set[1][4]).item()
                    mae_loss_10 += self.criterion(batch_set[0][9], batch_set[1][9]).item()
                    mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                    mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                    mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()

        self.performance_data.append([mae_loss/index, mae_loss_1/index, mae_loss_5/index, mae_loss_10/index, mae_loss_x/index,
                                mae_loss_y/index, mae_loss_z/index])


    def calc_train_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append(["test loss MAE(L1): ", (sum([i[0] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 1: ", (sum([i[1] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 5: ", (sum([i[2] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred ts 10: ", (sum([i[3] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force x: ", (sum([i[4] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force y: ", (sum([i[5] for i in self.performance_data]) / len(self.performance_data))])
        performance_data_full.append(["test loss MAE(L1) pred force z: ", (sum([i[6] for i in self.performance_data]) / len(self.performance_data))])

        self.objects = list(set(self.objects))
        for object in self.objects:
            performance_data_full.append(["test loss MAE(L1) Trained object " + str(object) + ": ", (sum([i[0] for i in self.performance_data if i[-1] == str(object)]) / len([i[0] for i in self.performance_data if i[-1] == str(object)]))])

        [print (i) for i in performance_data_full]
        # np.save(data_save_path + 'TRAIN_model_performance_loss_data', np.asarray(performance_data_full))

    def save_predictions(self, experiment_to_test):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''
        trial_groundtruth_data = []
        trial_predicted_data_t1 = []
        trial_predicted_data_t5 = []
        trial_predicted_data_t10 = []
        for index in range (len (self.tg_back_scaled)):
            for batch_number in range (len (self.tp10_back_scaled[index])):
                if experiment_to_test == self.prediction_data[index][2].T[batch_number][0]:
                    trial_predicted_data_t1.append (self.tp1_back_scaled[index][batch_number])
                    trial_predicted_data_t5.append (self.tp5_back_scaled[index][batch_number])
                    trial_predicted_data_t10.append (self.tp10_back_scaled[index][batch_number])
                    trial_groundtruth_data.append (self.tg_back_scaled[index][batch_number])

        plot_save_dir = data_save_path + "SCALED_train_plots_" + str (experiment_to_test)
        try:
            os.mkdir (plot_save_dir)
        except:
            "directory already exists"

        np.save(plot_save_dir + '/trial_groundtruth_data', np.array (trial_groundtruth_data))
        np.save(plot_save_dir + '/prediction_data_t1', np.array (trial_predicted_data_t1))
        np.save(plot_save_dir + '/prediction_data_t5', np.array (trial_predicted_data_t5))
        np.save(plot_save_dir + '/prediction_data_t10', np.array (trial_predicted_data_t10))


    def create_difference_gifs(self, experiment_to_test):
        '''
        - Create gifs showing the predicted images for a trial, the groundtruth and the difference between the two.
        - Only at t+10
        '''
        plot_save_dir = data_save_path + "test_plots_" + str (self.current_exp)
        try:
            os.mkdir (plot_save_dir)
        except:
            "directory already exists"

        ts_to_test = -1
        trial_groundtruth_data_t10 = []
        trial_predicted_data_t10 = []
        for index in range (len (self.prediction_data)):
            for batch_number in range (self.prediction_data[index][0].shape[1]):
                trial_predicted_data_t10.append (
                    self.prediction_data[index][0][ts_to_test][batch_number].permute (1, 2, 0).numpy ())
                trial_groundtruth_data_t10.append (
                    self.prediction_data[index][1][ts_to_test][batch_number].permute (1, 2, 0).numpy ())

        prediction_data_to_plot_images = np.array (trial_predicted_data_t10)[:]
        gt_data_to_plot_images = np.array (trial_groundtruth_data_t10)[:]
        images = [abs (prediction_data_to_plot_images[i] - gt_data_to_plot_images[i]) for i in
                  range (len (gt_data_to_plot_images))]

        for feature in range (3):
            image_player (gt_data_to_plot_images, "groundtruth_t10", feature, self.current_exp)
            image_player (prediction_data_to_plot_images, "predicted_t10", feature, self.current_exp)
            image_player (images, "gt10_and_pt10_difference", feature, self.current_exp)

    def create_test_plots(self, experiment_to_test):
        '''
        - Plot the descaled 48 feature tactile vector for qualitative analysis
        - Save plots in a folder with name being the trial number.
        '''
        trial_groundtruth_data = []
        trial_predicted_data_t10 = []
        trial_predicted_data_t5 = []
        for index in range (len (self.tg_back_scaled)):
            for batch_number in range (len (self.tp10_back_scaled[index])):
                if experiment_to_test == self.prediction_data[index][2].T[batch_number][0]:
                    trial_predicted_data_t10.append (self.tp10_back_scaled[index][batch_number])
                    trial_predicted_data_t5.append (self.tp5_back_scaled[index][batch_number])
                    trial_groundtruth_data.append (self.tg_back_scaled[index][batch_number])

        plot_save_dir = data_save_path + "test_plots_" + str (experiment_to_test)
        try:
            os.mkdir (plot_save_dir)
        except:
            "directory already exists"

        np.save (plot_save_dir + '/prediction_data_gt', np.array (self.tg_back_scaled))
        np.save (plot_save_dir + '/prediction_data_t5', np.array (self.tp5_back_scaled))
        np.save (plot_save_dir + '/prediction_data_t10', np.array (self.tp10_back_scaled))

        if len (trial_groundtruth_data) > 300 and len (trial_groundtruth_data) > 450:
            trim_min = 100
            trim_max = 250
        elif len (trial_groundtruth_data) > 450:
            trim_min = 200
            trim_max = 450
        else:
            trim_min = 0
            trim_max = -1

        index = 0
        titles = ["sheerx", "sheery", "normal"]
        for j in range (3):
            for i in range (16):
                groundtruth_taxle = []
                predicted_taxel_t10 = []
                predicted_taxel_t5 = []
                for k in range (len (trial_predicted_data_t10)):
                    predicted_taxel_t10.append (trial_predicted_data_t10[k][index])
                    predicted_taxel_t5.append (trial_predicted_data_t5[k][index])
                    groundtruth_taxle.append (trial_groundtruth_data[k][index])

                index += 1

                fig, ax1 = plt.subplots ()
                ax1.set_xlabel ('time step')
                ax1.set_ylabel ('tactile reading')
                line_1 = ax1.plot ([i for i in predicted_taxel_t10], alpha=1.0, c="b", label="Pred_t10")
                line_2 = ax1.plot ([i for i in predicted_taxel_t5], alpha=1.0, c="g", label="Pred_t5")
                line_3 = ax1.plot (groundtruth_taxle, alpha=1.0, c="r", label="Gt")
                ax1.tick_params (axis='y')

                ax2 = ax1.twinx ()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel ('MAE between gt+t10, pred_t10')  # we already handled the x-label with ax1
                line_4 = ax2.plot ([None for i in range (10)] + [abs (pred - gt) for (gt, pred) in
                                                                 zip (groundtruth_taxle[10:],
                                                                      predicted_taxel_t10[:-10])],
                                   alpha=1.0, c="k", label="MAE")

                lines = line_1 + line_2 + line_3 + line_4
                labs = [l.get_label () for l in lines]

                fig.tight_layout ()  # otherwise the right y-label is slightly clipped
                fig.subplots_adjust (top=0.90)
                ax1.legend (lines, labs, loc="upper right")
                if len (predicted_taxel_t5) < 150:
                    ax1.xaxis.set_major_locator (MultipleLocator (10))
                    ax1.xaxis.set_minor_locator (AutoMinorLocator (10))
                elif len (predicted_taxel_t5) > 150 and len (predicted_taxel_t5) < 1000:
                    ax1.xaxis.set_major_locator (MultipleLocator (25))
                    ax1.xaxis.set_minor_locator (AutoMinorLocator (25))
                elif len (predicted_taxel_t5) > 1000:
                    ax1.xaxis.set_major_locator (MultipleLocator (100))
                    ax1.xaxis.set_minor_locator (AutoMinorLocator (100))

                ax1.grid (which='minor')
                ax1.grid (which='major')
                plt.title ("AV-PMN model taxel " + str (index))
                plt.savefig (plot_save_dir + '/test_plot_taxel_' + str (index) + '.png', dpi=300)
                plt.clf ()

    def calc_trial_performance(self):
        mae_loss, mae_loss_1, mae_loss_5, mae_loss_10, mae_loss_x, mae_loss_y, mae_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ssim_loss, ssim_loss_1, ssim_loss_5, ssim_loss_10, ssim_loss_x, ssim_loss_y, ssim_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        psnr_loss, psnr_loss_1, psnr_loss_5, psnr_loss_10, psnr_loss_x, psnr_loss_y, psnr_loss_z = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        ssim_calc = SSIM (window_size=32)
        psnr_calc = PSNR ()


        index = 0
        index_ssim = 0
        with torch.no_grad ():
            for batch_set in self.prediction_data:
                index += 1

                mae_loss_check = self.criterion (batch_set[0], batch_set[1]).item ()
                if math.isnan (mae_loss_check):
                    index -= 1
                else:
                    ## MAE:
                    mae_loss += mae_loss_check
                    mae_loss_1 += self.criterion (batch_set[0][0], batch_set[1][0]).item ()
                    mae_loss_5 += self.criterion (batch_set[0][4], batch_set[1][4]).item ()
                    mae_loss_10 += self.criterion (batch_set[0][9], batch_set[1][9]).item ()
                    mae_loss_x  += self.criterion(batch_set[0][:,:,0], batch_set[1][:,:,0]).item()
                    mae_loss_y  += self.criterion(batch_set[0][:,:,1], batch_set[1][:,:,1]).item()
                    mae_loss_z  += self.criterion(batch_set[0][:,:,2], batch_set[1][:,:,2]).item()
                    ## SSIM:
                    for i in range (len (batch_set)):
                        index_ssim += 1
                        ssim_loss += ssim_calc (batch_set[0][i], batch_set[1][i])
                    ssim_loss_1 += ssim_calc (batch_set[0][0], batch_set[1][0])
                    ssim_loss_5 += ssim_calc (batch_set[0][4], batch_set[1][4])
                    ssim_loss_10 += ssim_calc (batch_set[0][9], batch_set[1][9])
                    ssim_loss_x += ssim_calc (batch_set[0][:, :, 0], batch_set[1][:, :, 0])
                    ssim_loss_y += ssim_calc (batch_set[0][:, :, 1], batch_set[1][:, :, 1])
                    ssim_loss_z += ssim_calc (batch_set[0][:, :, 2], batch_set[1][:, :, 2])
                    ## PSNR:
                    psnr_loss += psnr_calc (batch_set[0], batch_set[1])
                    psnr_loss_1 += psnr_calc (batch_set[0][0], batch_set[1][0])
                    psnr_loss_5 += psnr_calc (batch_set[0][4], batch_set[1][4])
                    psnr_loss_10 += psnr_calc (batch_set[0][9], batch_set[1][9])
                    psnr_loss_x += psnr_calc (batch_set[0][:, :, 0], batch_set[1][:, :, 0])
                    psnr_loss_y += psnr_calc (batch_set[0][:, :, 1], batch_set[1][:, :, 1])
                    psnr_loss_z += psnr_calc (batch_set[0][:, :, 2], batch_set[1][:, :, 2])

        self.performance_data.append (
            [mae_loss / index, mae_loss_1 / index, mae_loss_5 / index, mae_loss_10 / index,
             mae_loss_x / index,
             mae_loss_y / index, mae_loss_z / index, ssim_loss / index_ssim, ssim_loss_1 / index,
             ssim_loss_5 / index, ssim_loss_10 / index, ssim_loss_x / index, ssim_loss_y / index,
             ssim_loss_z / index, psnr_loss / index, psnr_loss_1 / index, psnr_loss_5 / index,
             psnr_loss_10 / index, psnr_loss_x / index, psnr_loss_y / index, psnr_loss_z / index])

    def calc_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        if len(self.performance_data) != 0:
            performance_data_full = []
            performance_data_full.append (["test loss MAE(L1): ", (
                        sum ([i[0] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss MAE(L1) pred ts 1: ", (
                        sum ([i[1] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss MAE(L1) pred ts 5: ", (
                        sum ([i[2] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss MAE(L1) pred ts 10: ", (
                        sum ([i[3] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss MAE(L1) pred force x: ", (
                        sum ([i[4] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss MAE(L1) pred force y: ", (
                        sum ([i[5] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss MAE(L1) pred force z: ", (
                        sum ([i[6] for i in self.performance_data]) / len (self.performance_data))])

            performance_data_full.append (["test loss MAE(L1) Seen objects: ", (
                        sum ([i[0] for i in self.performance_data if i[-2] == '1']) / len (
                    [i[0] for i in self.performance_data if i[-2] == '1']))])
            performance_data_full.append (["test loss MAE(L1) Novel objects: ", (
                        sum ([i[0] for i in self.performance_data if i[-2] == '0']) / len (
                    [i[0] for i in self.performance_data if i[-2] == '0']))])


            performance_data_full.append (["test loss SSIM: ", (
                        sum ([i[7] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss SSIM pred ts 1: ", (
                        sum ([i[8] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss SSIM pred ts 5: ", (
                        sum ([i[9] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss SSIM pred ts 10: ", (
                        sum ([i[10] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss SSIM pred force x: ", (
                        sum ([i[11] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss SSIM pred force y: ", (
                        sum ([i[12] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss SSIM pred force z: ", (
                        sum ([i[13] for i in self.performance_data]) / len (self.performance_data))])

            performance_data_full.append (["test loss SSIM Seen objects: ", (
                        sum ([i[7] for i in self.performance_data if i[-2] == '1']) / len (
                    [i[7] for i in self.performance_data if i[-2] == '1']))])
            performance_data_full.append (["test loss SSIM Novel objects: ", (
                        sum ([i[7] for i in self.performance_data if i[-2] == '0']) / len (
                    [i[7] for i in self.performance_data if i[-2] == '0']))])

            performance_data_full.append (["test loss PSNR: ", (
                        sum ([i[14] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss PSNR pred ts 1: ", (
                        sum ([i[15] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss PSNR pred ts 5: ", (
                        sum ([i[16] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss PSNR pred ts 10: ", (
                        sum ([i[17] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss PSNR pred force x: ", (
                        sum ([i[18] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss PSNR pred force y: ", (
                        sum ([i[19] for i in self.performance_data]) / len (self.performance_data))])
            performance_data_full.append (["test loss PSNR pred force z: ", (
                        sum ([i[20] for i in self.performance_data]) / len (self.performance_data))])

            performance_data_full.append (["test loss PSNR Seen objects: ", (
                        sum ([i[14] for i in self.performance_data if i[-2] == '1']) / len (
                    [i[14] for i in self.performance_data if i[-2] == '1']))])
            performance_data_full.append (["test loss PSNR Novel objects: ", (
                        sum ([i[14] for i in self.performance_data if i[-2] == '0']) / len (
                    [i[14] for i in self.performance_data if i[-2] == '0']))])

            [print (i) for i in performance_data_full]
            np.save (data_save_path + 'model_performance_loss_data', np.asarray (performance_data_full))

    def load_scalars(self):
        self.side_cam_std_scaler_all = []
        self.side_cam_minmax_scaler_all = []

        self.scaler_side_cam_r = load (open (scaler_dir + "side_cam_standard_scaler_r.pkl", 'rb'))
        self.scaler_side_cam_g = load (open (scaler_dir + "side_cam_standard_scaler_g.pkl", 'rb'))
        self.scaler_side_cam_b = load (open (scaler_dir + "side_cam_standard_scaler_b.pkl", 'rb'))
        self.min_max_scalerr_full_data = load (open (scaler_dir + "side_cam_min_max_scaler_r.pkl", 'rb'))
        self.min_max_scalerg_full_data = load (open (scaler_dir + "side_cam_min_max_scaler_g.pkl", 'rb'))
        self.min_max_scalerb_full_data = load (open (scaler_dir + "side_cam_min_max_scaler_b.pkl", 'rb'))

        # Append the rgb into the all. The following two will be a list of the 3 elements
        scal_arr = [self.scaler_side_cam_r, self.scaler_side_cam_g, self.scaler_side_cam_b]
        scal_minmax_arr = [self.min_max_scalerr_full_data, self.min_max_scalerg_full_data, self.min_max_scalerb_full_data]
        for elems in scal_arr:
            self.side_cam_std_scaler_all.append(elems)
        for elems in scal_minmax_arr:
            self.side_cam_minmax_scaler_all.append(elems)
        

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean ((img1 - img2) ** 2)
        return 20 * torch.log10 (255.0 / torch.sqrt (mse))

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor (
        [exp (-(x - window_size // 2) ** 2 / float (2 * sigma ** 2)) for x in range (window_size)])
    return gauss / gauss.sum ()

def create_window(window_size, channel):
    _1D_window = gaussian (window_size, 1.5).unsqueeze (1)
    _2D_window = _1D_window.mm (_1D_window.t ()).float ().unsqueeze (0).unsqueeze (0)
    window = Variable (_2D_window.expand (channel, 1, window_size, window_size).contiguous ())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d (img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d (img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow (2)
    mu2_sq = mu2.pow (2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d (img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d (img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d (img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean ()
    else:
        return ssim_map.mean (1).mean (1).mean (1)

class SSIM (torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super (SSIM, self).__init__ ()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window (window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size ()

        if channel == self.channel and self.window.data.type () == img1.data.type ():
            window = self.window
        else:
            window = create_window (self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda (img1.get_device ())
            window = window.type_as (img1)

            self.window = window
            self.channel = channel

        return _ssim (img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size ()
    window = create_window (window_size, channel)

    if img1.is_cuda:
        window = window.cuda (img1.get_device ())
    window = window.type_as (img1)

    return _ssim (img1, img2, window, window_size, channel, size_average)

if __name__ == "__main__":
    BG = BatchGenerator ()
    MT = ModelTester ()
    MT.test_full_model ()
    MT.calc_test_performance ()