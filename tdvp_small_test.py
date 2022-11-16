
import csv
import glob
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from tdvp_big import ACTVP

class BatchGenerator:
    def __init__(self, batch_size, test_directory):
        self.batch_size = batch_size
        self.data_map = []
        self.test_directory = test_directory
        val = int(test_directory[-1]) + 34
        # val = test_directory[-1]
        with open(test_directory + '/map_' + str(val) + '.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map, self.test_directory)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False)
        self.data_map = []
        return test_loader

class FullDataSet:
    def __init__(self, data_map, test_direc):
        self.test_directory = test_direc
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_task_data = np.load(self.test_directory + "/" + value[0])
        robot_joint_data = np.load(self.test_directory + "/" + value[1])

        side_camera_image_names = []
        for image_name in np.load(self.test_directory + "/" + value[3]):
            side_camera_image_names.append(np.load(self.test_directory + "/" + image_name))

        experiment_number = np.load(self.test_directory + "/" +value[4])
        time_steps = np.load(self.test_directory + "/" + value[5])
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
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device).to(device))

class Test:
    def __init__(self, model_save_path):
        self.batch_size = 32
        self.current_exp = 0
        self.index = 0

        self.performance_data_all = []

        # self.model = ACTVP()
        self.model = torch.load(model_save_path)
        # self.model.load_state_dict(saved_model)
        self.criterion = nn.L1Loss()

        experiment_number_files = glob.glob(exp_test_dir + "/*") 
        for index, directory in tqdm(enumerate(experiment_number_files)):
            
            BG = BatchGenerator(batch_size = self.batch_size, test_directory=directory)
            self.test_full_loader = BG.load_full_data()
        
            self.test_model()
            self.calculate_test_performance()

    def test_model(self):
        batch_losses = []
        self.performance_data = []

        for index, batch_features in enumerate(self.test_full_loader):
            if batch_features[2].shape[0] == self.batch_size:   # side_camera_data
                groundtruth_scene, predictions_scene = self.format_and_run_batch(batch_features, test=True) 

                # Quantitative analysis
                # batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[context_frames:]))

                # # Qualitative analysis
                # Context of 5 to 14 across all experiments. 
                print(type(int(batch_features[4][0][5].item())))
                if (int(batch_features[4][0][5].item() )) == int(5): # and batch_features[5][0] == 25
                    print("In here")
                    self.save_images(batch_features, groundtruth_scene[context_frames:], predictions_scene)
                
                self.prediction_data = []
    

    def format_and_run_batch(self, batch_features, test):
        
        mae, kld, mae_side_cam, predictions = 100, 100, 100, 100
        self.prediction_data = []
        side_camera = batch_features[2].permute(1, 0, 4, 3, 2).to(device)
        robot_joint = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
        
        side_camera_predictions = self.model.forward(side_camera=side_camera,
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

        # for i in [0, 1]:
        #     if i == 0:
        #         side_camera_cut = side_camera[:, 0:current_batch, :, :, :]
        #         side_camera_predictions_cut = side_camera_predictions[:, 0:current_batch, :, :, :]
        #         experiment_number_cut = experiment_number[0:current_batch]
        #         time_steps_cut = time_steps[:, 0:current_batch]
        #     if i == 1:
        side_camera_cut = side_camera[:, current_batch:, :, :, :]
        side_camera_predictions_cut = side_camera_predictions[:, current_batch:, :, :, :]
        experiment_number_cut = experiment_number[current_batch:]
        time_steps_cut = time_steps[:, current_batch:]

        self.prediction_data.append(
            [side_camera_predictions_cut.cpu().detach(), side_camera_cut[context_frames:].cpu().detach(),
                experiment_number_cut.cpu().detach(), time_steps_cut.cpu().detach()])

        print ("currently testing trial number: ", str (self.current_exp))
        # self.calc_train_trial_performance()
            
        self.calc_trial_preformance()
        # self.save_predictions(self.current_exp)
        # self.create_test_plots(self.current_exp)
        # self.create_difference_gifs(self.current_exp)
        
        self.current_exp += 1
    
        return side_camera, self.prediction_data

    def calc_trial_preformance(self):
        mae_loss = 0.0
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

        self.performance_data.append (mae_loss / index)

    def calculate_test_performance(self):
        '''
        - Calculates PSNR, SSIM, MAE for ts1, 5, 10 and x,y,z forces
        - Save Plots for qualitative analysis
        - Slip classification test
        '''
        performance_data_full = []
        performance_data_full.append (["test loss MAE(L1): ", (
                    sum (self.performance_data) / len (self.performance_data))])
        self.performance_data_all.append(sum (self.performance_data) / len (self.performance_data))

        [print (i) for i in performance_data_full]
        np.save (data_save_path + 'model_performance_loss_data', np.asarray (self.performance_data_all))


    def calculate_scores(self, prediction_scene, groundtruth_scene, prediction_tactile=None, groundtruth_tactile=None):
        scene_losses_full, scene_losses_last = [],[]
        for criterion in [nn.L1Loss(), nn.MSELoss()]:  #, SSIM(window_size=self.image_width)]:
            scene_batch_loss_full = []
            for i in range(prediction_scene.shape[0]):
                scene_batch_loss_full.append(criterion(prediction_scene[i], groundtruth_scene[i]).cpu().detach().data)

            scene_losses_full.append(sum(scene_batch_loss_full) / len(scene_batch_loss_full))
            scene_losses_last.append(criterion(prediction_scene[-1], groundtruth_scene[-1]).cpu().detach().data)  # t+5

        return [scene_losses_full, scene_losses_last]


    def save_images(self, batch_features, side_camera, side_camera_predictions):

        for index, batch in enumerate(batch_features):

            experiment = str(batch[4][0])
            time_step = str(batch[5][0])

            for image_gt, image_pred in zip(side_camera[index][context_frames:], side_camera_predictions[index]):
                im = Image.fromarray(image_gt)
                im.save("imageGT_ " + experiment  + " time_step_" +  time_step + ".jpeg")

                im = Image.fromarray(image_pred)
                im.save("imagePRED_ " + experiment  + " time_step_" +  time_step + ".jpeg")
            
                # save numpy:

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available
    context_frames = 5

    data_save_path  = "/home/venkatesh/tdvp/teleop_data/models/THDloss_model_07_11_2022_14_41/test/"
    model_save_path = "/home/venkatesh/tdvp/teleop_data/models/THDloss_model_07_11_2022_14_41/ACTVP_THD"
    # test_data_dir   = "/home/venky/time_delay_vp/test_out/test_trial_0/"
    exp_test_dir = "/home/venkatesh/tdvp/teleop_data/test_out/"

    t = Test(model_save_path)
    # t.test_model()