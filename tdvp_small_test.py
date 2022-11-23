
import csv
import glob
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from tdvp import ACTVP

class BatchGenerator:
    def __init__(self):
        # self.batch_size = batch_size
        self.data_map = []
        # self.test_directory = test_directory
        # val = int(test_data_dir[-2]) + 34
        # val = test_directory[-1]
        with open(exp_test_dir + 'map.csv', 'r') as f:  # rb
            reader = csv.reader(f)
            for row in reader:
                self.data_map.append(row)

    def load_full_data(self):
        dataset_test = FullDataSet(self.data_map)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
        self.data_map = []
        # Display image.
        # for index, batch_features in enumerate(test_loader):
        #     # batch_features[2] is the tensor containing side camera data of shape 32,7,32,32,3
        #     side_camera = batch_features[2]
        #     img = side_camera[0][0]    # First batch, first frame of sequence
        #     img = img.cpu().detach().numpy()
        #     im = Image.fromarray((img*255).astype(np.uint8))
        #     im.save("Image_" + str(index) + str(batch) +".jpeg")
        return test_loader

class FullDataSet:
    def __init__(self, data_map):
        self.samples = data_map[1:]
        data_map = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        value = self.samples[idx]
        robot_task_data = np.load(exp_test_dir + value[0])
        robot_joint_data = np.load(exp_test_dir + value[1])

        side_camera_image_names = []
        for image_name in np.load(exp_test_dir + value[3]):
            side_camera_image_names.append(np.load(exp_test_dir + image_name))

        experiment_number = np.load(exp_test_dir +value[4])
        time_steps = np.load(exp_test_dir + value[5])
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
        self.batch_size = batch_size
        self.current_exp = 0
        self.index = 0

        self.test_full_loader = BG.load_full_data()
        # self.model = ACTVP()
        self.model = torch.load(model_save_path)
        # self.model.load_state_dict(saved_model)
        self.criterion = nn.L1Loss()

        # experiment_number_files = glob.glob(exp_test_dir + "/*") 
        # for index, directory in enumerate(exp_test_dir):
            
        #     BG = BatchGenerator(batch_size = self.batch_size, test_directory=directory)
        #     self.test_full_loader = BG.load_full_data()
        
        self.test_model()
        self.calculate_test_performance()

    def test_model(self):
        self.performance_data = []
        self.save_performance_data = []
        test_loss= []
        val = True

        progress_bar = tqdm(range(0, epochs), total=(epochs*len(self.test_full_loader)))
        for epoch in progress_bar:
            for index, batch_features in enumerate(self.test_full_loader):
                
                # self.test_image(batch_features)
                groundtruth_scene, predictions_scene = self.format_and_run_batch(batch_features, index)
                # Saving the image
                if val == True:
                    self.test_image(groundtruth_scene, index=index, modify_img_shape=True, gt=True)
                    self.test_image(predictions_scene, index=index, modify_img_shape=True, prediction=True)
                    val = False
                self.calculate_test_performance() 
                # print("Here")

                # Quantitative analysis
                # batch_losses.append(self.calculate_scores(predictions_scene, groundtruth_scene[context_frames:]))

                # # Qualitative analysis
                # print(type(int(batch_features[4][0][5].item())))
                # Looking at time steps 5 to 14 across all experiments and saving them
                if (int(batch_features[4][0][5].item() )) == int(5): # and batch_features[5][0] == 25
                    # print("In here")
                    pass
                    # self.save_images(batch_features, groundtruth_scene, predictions_scene)
                

                # if batch_features[4] == 5 and batch_features[5][0] == 25:   
                #     self.save_images(predictions_scene, groundtruth_scene[context_frames:], index)
            progress_bar.update()

            test_loss.append( np.mean(self.save_performance_data))
            self.save_performance_data = []
        np.save (data_save_path + 'model_performance_loss_data_all', np.asarray (test_loss))
    

    def test_image(self, batch_features, index = None, modify_img_shape=False, prediction=False, gt = False):
        # side_camera = batch_features[2].to(device)
        side_camera_cut = batch_features

        # Perform this in post processing
        img = side_camera_cut[0][0]    # 0th frame in sequence, 0th batch
        img = img.cpu().detach().numpy()
        if modify_img_shape:
            img = np.transpose(img, (1,2,0))  # Image to be 32, 32, 3
        img = np.rot90(np.fliplr(img))
        im = Image.fromarray((img*255).astype(np.uint8)).convert("RGB")
        # im.save("Image_gt.jpeg")
        if prediction == True:
            im.save("Image_pred" + str(index) + ".jpeg")
        if gt == True:
            im.save("Image_gt" + str(index) + ".jpeg")

    def pad_data(self, batch_features):
        dim_task_space = batch_features[0].shape
        dim_joint_space = batch_features[1].shape
        dim_side_cam = batch_features[2].shape
        dim_exp = batch_features[3].shape
        dim_time = batch_features[4].shape

        zeros_task_data = torch.zeros(self.batch_size - len(batch_features[0]), dim_task_space[1], dim_task_space[2])
        zeros_joint_data = torch.zeros(self.batch_size - len(batch_features[1]), dim_joint_space[1], dim_joint_space[2])
        zeros_side_cam = torch.zeros(self.batch_size - len(batch_features[2]), dim_side_cam[1], dim_side_cam[2], dim_side_cam[3], dim_side_cam[4])
        zeros_exp = torch.zeros(self.batch_size - len(batch_features[3]))
        zeros_time = torch.zeros(self.batch_size - len(batch_features[4]), dim_time[1])

        # Pad with zeros to make length same as batch size
        robot_task = torch.cat((batch_features[0], zeros_task_data ))
        robot_joint = torch.cat((batch_features[1], zeros_joint_data ))
        side_camera = torch.cat((batch_features[2], zeros_side_cam))
        exp = torch.cat((batch_features[3], zeros_exp))
        time_data = torch.cat((batch_features[4], zeros_time))

        return robot_task, robot_joint, side_camera, exp, time_data


    def format_and_run_batch(self, batch_features, index):
        
        mae, kld, mae_side_cam, predictions = 100, 100, 100, 100
        self.prediction_data = []
        current_batch = len(batch_features[0])
        if len(batch_features[2]) == self.batch_size:   # side_camera_data
            side_camera = batch_features[2].permute(1, 0, 4, 3, 2).to(device)
            robot_task = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            i = 1

            # self.test_image(batch_features)
        else:
            
            robot_task, robot_joint, side_camera, experiment_number, time_steps = self.pad_data(batch_features)
            
            side_camera = batch_features[2].permute(1, 0, 4, 3, 2).to(device)
            robot_task = batch_features[0].squeeze(-1).permute(1, 0, 2).to(device)
            i = 0

        side_camera_predictions = self.model.forward(side_camera=side_camera,
                                                        robot_task_data=robot_task)  # Step 3. Run our forward pass.
        
        if i == 0:
            side_camera_cut = side_camera[:, 0:current_batch, :, :, :]
            side_camera_predictions_cut = side_camera_predictions[:, 0:current_batch, :, :, :]
        if i == 1:
            side_camera_cut = side_camera[:, :, :, :, :]
            side_camera_predictions_cut = side_camera_predictions[:, :, :, :, :]

        # Saving the image
        # self.test_image(side_camera_predictions_cut, index=index, modify_img_shape=True)

        self.prediction_data.append(
            [side_camera_cut[context_frames:].cpu().detach(), side_camera_predictions_cut.cpu().detach()])

        # print ("currently testing trial number: ", str (self.current_exp))
        # self.calc_train_trial_performance()
            
        self.calc_trial_preformance()
        # self.save_predictions(self.current_exp)
        # self.create_test_plots(self.current_exp)
        # self.create_difference_gifs(self.current_exp)
        # self.prediction_data = []

        # Takes the value of experiment_number which is a tensor with all 0s or 1s etc.. self.curent_exp should be same as the experiment_number for current_batch to imcrement
        self.current_exp += 1   
    
        # returning grouth truth and predictions
        return self.prediction_data[0][0], self.prediction_data[0][1]

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

        # [print (i) for i in performance_data_full]
        self.save_performance_data.append(sum (self.performance_data) / len (self.performance_data))
        np.save (data_save_path + 'model_performance_loss_data', np.asarray (performance_data_full))


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

        # for index, batch in enumerate(batch_features):
        for index in range(len(batch_features)):

            experiment = str(int(batch_features[3][0].item()))
            time_step = str(int(batch_features[4][0][0].item()))

            gt = side_camera[index][index].cpu().detach().numpy()
            gt = np.transpose(gt, (1,2,0))
            im = Image.fromarray((gt * 255).astype(np.uint8)).convert("RGB")
            im.save("imageGT2_ " + experiment  + " time_step_" +  time_step + ".jpeg")

            pred = side_camera_predictions[index][index].cpu().detach().numpy()
            pred = np.transpose(pred, (1,2,0))
            im = Image.fromarray((pred * 255).astype(np.uint8)).convert("RGB")
            im.save("imagePRED2_ " + experiment  + " time_step_" +  time_step + ".jpeg")

            # for image_gt, image_pred in zip(side_camera[index][context_frames:], side_camera_predictions[index]):
            #     print("Saving Image")
            #     im = Image.fromarray(image_gt)
            #     im.save("imageGT_ " + experiment  + " time_step_" +  time_step + ".jpeg")

            #     im = Image.fromarray(image_pred)
            #     im.save("imagePRED_ " + experiment  + " time_step_" +  time_step + ".jpeg")
            
                # save numpy:

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#  use gpu if available
    context_frames = 5
    batch_size = 32
    epochs= 50

    # Uncomment if using laptop
    # data_save_path  = "/home/venky/time_delay_vp/models/THDloss_model_18_11_2022_11_17/scaled_test/"
    # model_save_path = "/home/venky/time_delay_vp/models/THDloss_model_18_11_2022_11_17/ACTVP_THD"
    # exp_test_dir   = "/home/venky/time_delay_vp/test_out/"

    # Uncomment if using cluster
    data_save_path  = "/home/venkatesh/tdvp_all/teleop_data/models/THDloss_model_22_11_2022_15_52/test/"
    model_save_path = "/home/venkatesh/tdvp_all/teleop_data/models/THDloss_model_22_11_2022_15_52/ACTVP_THD"
    exp_test_dir = "/home/venkatesh/tdvp_all/teleop_data/test_out/"

    BG = BatchGenerator()
    t = Test(model_save_path)
    # t.test_model()