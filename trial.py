import numpy as np
import cv2
import matplotlib.pyplot as  plt

base_path = "/home/venky/time_delay_vp/"
train_path = "/train"
test_path = "/test"

test_out = "/test_out"

img_rows = 32
img_cols = 32

def load_data(exp_no):
    arr = np.load(base_path + train_path + "/exp" + str(exp_no) + "/top_camera_" + str(exp_no) + ".npy")
    print(type(arr))
    return arr

def load_data_out(exp_no):
    exp_arr = np.load(base_path + test_out + "/test_trial_0" + "/experiment_number_" + str(exp_no) + ".npy") 
    robot_data_arr = np.load(base_path + test_out + "/test_trial_0" + "/robot_data_euler_" + str(exp_no) + ".npy" )
    return exp_arr, robot_data_arr

def resize(exp_no):
    path=base_path + train_path + "/exp" + str(exp_no) + "/images_tc/image_" + str(exp_no) + ".jpeg"
    img=cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (img_rows, img_cols))
    return 0

def load_loss(path, training=False, validation=False):
    if training:
        arr = np.load(path + "plot_training_loss.npy")
    if validation:
        arr = np.load(path + "plot_validation_loss.npy")
    return arr

arr = load_data(1)
resize(1)
# load_data_out(1)
array = load_loss("/home/venky/time_delay_vp/models/THDloss_model_03_11_2022_15_51/", validation=True)
# print(array.shape)
x = range(len(array))
plt.plot(x, array)
plt.show()
