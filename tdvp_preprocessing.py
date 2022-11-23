
import os
import csv
import cv2
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pickle import dump

from sklearn import preprocessing
from scipy.spatial.transform import Rotation as R

side_image = True
image_height = 32
image_width = 32
context_length = 5
horrizon_length = 2

# Uncomment following if working on laptop
# base_path = "/home/venky/time_delay_vp/"
# train_data_dir = base_path + "train/"
# test_data_dir = base_path + "test/"
# scaler_out_dir = base_path + 'scalar_info_universal/'

# train_out_dir = base_path + "train_out/"
# test_out_dir = base_path + "test_out/"

# Uncomment following if working on cluster
base_path = "/media/venkatesh/Extreme SSD/pilot_object_data/push_object_data/"
train_data_dir = base_path + "train/"
test_data_dir = base_path + "test/"

save_path = "/home/venkatesh/tdvp_all/teleop_data/"
scaler_out_dir = save_path + 'scaler_info_universal/'
train_out_dir = save_path + "train_out/"
test_out_dir = save_path + "test_out/"



class data_formatter:
    def __init__(self):
        self.files_train = []
        self.files_test = []
        self.full_data_robot_task = []
        self.full_data_robot_joint = []
        self.full_side_cam_data = []

        self.side_image_names = []
        self.all_reshaped = []
        self.side_image = side_image
        self.image_height = image_height
        self.image_width = image_width
        self.context_length = context_length
        self.horrizon_length = horrizon_length

    def load_file_names(self):
        self.files_train = glob.glob(train_data_dir + '/*')
        self.files_test = glob.glob(test_data_dir + '/*')

    def scale_data(self):
        files = self.files_train + self.files_test
        for exp_no, file in tqdm(enumerate(files), desc="Scaling data"):
            robot_task, robot_joint, side_camera, meta = self.load_file_data(file, exp_no+1)
            self.full_data_robot_task += list(robot_task)
            self.full_data_robot_joint += list(robot_joint)

        self.full_data_robot_task = np.array(self.full_data_robot_task)   # Gets the data of task space from all experiments
        self.full_data_robot_joint = np.array(self.full_data_robot_joint)
        
        # Scales the data to range (0,1), basically normalizing all features (pose) across all experiments
        self.robot_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot_task[:, feature].reshape(-1, 1)) for feature in range(6)]
        self.robot_joint_min_max_scalar = [preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(self.full_data_robot_joint[:, feature].reshape(-1, 1)) for feature in range(7)]

        # Standardize images
        # Scaling the data channel-wise
        # for i in tqdm(range(len(self.full_side_cam_data.reshape(-1,1)))):
        #     self.full_side_cam_data_scaled = self.full_side_cam_data[i] / 255

        self.save_scalars()


    def create_map(self):
        for stage in [train_out_dir, test_out_dir]:
            self.path_file = []
            index_to_save = 0
            print(stage)
            if stage == train_out_dir:
                files_to_run = self.files_train
            else:
                files_to_run = self.files_test

            for experiment_number, file in tqdm(enumerate(files_to_run), desc = "Creating map"):
                path_save = stage

                if stage != train_out_dir:
                    experiment_number += 33 
                robot_task, robot_joint, side_camera, meta = self.load_file_data(file, experiment_number+1)

                # scale the data

                for index, min_max_scalar in enumerate(self.robot_min_max_scalar):
                    robot_task[:, index] = np.squeeze(min_max_scalar.transform(robot_task[:, index].reshape(-1, 1)))

                for index, min_max_scalar in enumerate(self.robot_joint_min_max_scalar):
                    robot_joint[:, index] = np.squeeze(min_max_scalar.transform(robot_joint[:, index].reshape(-1, 1)))

                reshaped_side_image_names = []
                if self.side_image:
                    for time_step in range(len(side_camera)):
                        image_name = "side_image_reshaped_" + str(experiment_number) + "_time_step_" + str(time_step) + ".npy"
                        reshaped_side_image_names.append(image_name)
                        np.save(path_save + image_name, side_camera[time_step])

                sequence_length = self.context_length + self.horrizon_length

                # len(robot_joint) is placed for this exp, create a variable that takes the minimum of robot_joint amd robot_task
                iter_length = np.min([len(robot_joint), len(robot_task), len(side_camera)])
                for time_step in range(iter_length - sequence_length):
                    robot_data_euler_sequence = [robot_task[time_step + t] for t in range(sequence_length)]
                    robot_data_joint_sequence = [robot_joint[time_step + t] for t in range(sequence_length)]
                    side_image_data_sequence     = [side_camera[time_step + t] for t in range(sequence_length)]
                    experiment_data_sequence  = experiment_number
                    time_step_data_sequence   = [time_step + t for t in range(sequence_length)]
                    if self.side_image:
                        reshaped_side_image_name_sequence = [reshaped_side_image_names[time_step + t] for t in range(sequence_length)]

                    ####################################### Save the data and add to the map ###########################################
                    np.save(path_save + 'robot_data_euler_' + str(index_to_save), robot_data_euler_sequence)
                    np.save(path_save + 'robot_data_joint_' + str(index_to_save), robot_data_joint_sequence)
                    np.save(path_save + 'side_image_reshaped_data_sequence_' + str(index_to_save), side_image_data_sequence)
                    if self.side_image:
                        np.save(path_save + 'side_image_reshaped_name_sequence_' + str(index_to_save), reshaped_side_image_name_sequence)
                    np.save(path_save + 'experiment_number_' + str(index_to_save), experiment_data_sequence)
                    np.save(path_save + 'time_step_data_' + str(index_to_save), time_step_data_sequence)
                    np.save(path_save + 'trial_meta_' + str(index_to_save), np.array(meta))

                    ref = []
                    ref.append('robot_data_euler_' + str(index_to_save) + '.npy')
                    ref.append('robot_data_joint_' + str(index_to_save) + '.npy')
                    ref.append('side_image_reshaped_data_sequence_' + str(index_to_save) + '.npy')
                    if self.side_image:
                        ref.append('side_image_reshaped_name_sequence_' + str(index_to_save) + '.npy')
                    ref.append('experiment_number_' + str(index_to_save) + '.npy')
                    ref.append('time_step_data_' + str(index_to_save) + '.npy')
                    ref.append('trial_meta_' + str(index_to_save) + '.npy')
                    self.path_file.append(ref)
                    index_to_save += 1


            self.save_map(path_save)

    def save_map(self, path, test=False):
        if test:
            with open(path + '/map_' + str(self.test_no) + '.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path_euler', 'robot_data_path_joint', 'side_image_data_sequence', 'side_image_name_sequence', 'experiment_number', 'time_steps', 'meta'])
                for row in self.path_file:
                    writer.writerow(row)
        else:
            with open(path + '/map.csv', 'w') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(['robot_data_path_euler', 'robot_data_path_joint','side_image_data_sequence', 'side_image_name_sequence', 'experiment_number', 'time_steps', 'meta'])
                for row in self.path_file:
                    writer.writerow(row)

    def load_file_data(self, file, exp_no):
        robot_task_state = np.load(file + '/task_space_' + str(exp_no) + '.npy')
        robot_joint_state = np.load(file + '/joint_states_' + str(exp_no) + '.npy')
        side_camera = self.reshape_image(file, exp_no)
        # side_camera = np.load(file + '/side_camera_' + file[-1] + '.npy')
        # convert orientation to euler:
        robot_task_space = np.array([[state[-3], state[-2], state[-1]] + list(R.from_quat([state[-7], state[-6], state[-5], state[-4]]).as_euler('zyx', degrees=True)) for state in robot_task_state[1:]]).astype(float)
        meta_data = np.load(file + '/meta_' + str(exp_no) + '.npy')

        return robot_task_space, robot_joint_state, side_camera, meta_data

    
    def reshape_image(self, file, exp_no):
        all_reshaped = []
        # for exp_no in range(1, total_exps+1):
        try:
            sc = np.load(file + '/side_camera_' + str(exp_no) + '.npy')
        except FileNotFoundError as e:
            print("No file exists: ", e)
        for time_step in range(sc.shape[0]):
            img = Image.fromarray(sc[time_step], 'RGB')
            opencv_img = np.array(img)
            reshaped_image = cv2.resize(opencv_img, (self.image_height, self.image_width))
            reshaped_image = reshaped_image/255 # Normalize data to 0-1; it is now in range 0-255
            all_reshaped.append(np.array(reshaped_image))
            # if self.side_image:
            #     image_name = "side_image_" + str(exp_no) + "_time_step_" + str(time_step) + ".npy"
            #     self.side_image_names.append(image_name)
        all_reshaped = np.array(all_reshaped)
        return all_reshaped

    def save_scalars(self):
        # save the scalars
        dump(self.robot_min_max_scalar[0], open(scaler_out_dir + 'robot_min_max_scalar_px.pkl', 'wb'))
        dump(self.robot_min_max_scalar[1], open(scaler_out_dir + 'robot_min_max_scalar_py.pkl', 'wb'))
        dump(self.robot_min_max_scalar[2], open(scaler_out_dir + 'robot_min_max_scalar_pz.pkl', 'wb'))
        dump(self.robot_min_max_scalar[3], open(scaler_out_dir + 'robot_min_max_scalar_ex.pkl', 'wb'))
        dump(self.robot_min_max_scalar[4], open(scaler_out_dir + 'robot_min_max_scalar_ey.pkl', 'wb'))
        dump(self.robot_min_max_scalar[5], open(scaler_out_dir + 'robot_min_max_scalar_ez.pkl', 'wb'))

        dump(self.robot_joint_min_max_scalar[0], open(scaler_out_dir + 'robot_joint_min_max_scalar_j1.pkl', 'wb'))
        dump(self.robot_joint_min_max_scalar[1], open(scaler_out_dir + 'robot_joint_min_max_scalar_j2.pkl', 'wb'))
        dump(self.robot_joint_min_max_scalar[2], open(scaler_out_dir + 'robot_joint_min_max_scalar_j3.pkl', 'wb'))
        dump(self.robot_joint_min_max_scalar[3], open(scaler_out_dir + 'robot_joint_min_max_scalar_j4.pkl', 'wb'))
        dump(self.robot_joint_min_max_scalar[4], open(scaler_out_dir + 'robot_joint_min_max_scalar_j5.pkl', 'wb'))
        dump(self.robot_joint_min_max_scalar[5], open(scaler_out_dir + 'robot_joint_min_max_scalar_j6.pkl', 'wb'))
        dump(self.robot_joint_min_max_scalar[6], open(scaler_out_dir + 'robot_joint_min_max_scalar_j7.pkl', 'wb'))


df = data_formatter()
df.load_file_names()
df.scale_data()
df.create_map()