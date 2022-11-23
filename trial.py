import numpy as np
import cv2
import matplotlib.pyplot as  plt
from PIL import Image
import PIL.ImageOps

base_path = "/home/venkatesh/tdvp_all/teleop_data"
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

def loss_model():
    test_arr = np.load("/home/venky/time_delay_vp/models/THDloss_model_04_11_2022_16_04/scaled_test/model_performance_loss_data_all.npy")
    train_arr = np.load("/home/venky/time_delay_vp/models/THDloss_model_04_11_2022_16_04/scaled_test/plot_training_loss.npy")
    val_arr = np.load("/home/venky/time_delay_vp/models/THDloss_model_04_11_2022_16_04/scaled_test/plot_validation_loss.npy")
    return test_arr, train_arr, val_arr

def flip_image():
    for i in range(362, 439):

        image = Image.open('Image_pred' + str(i) + '.jpeg')

        inverted_image = PIL.ImageOps.invert(image)

        inverted_image.save('Image_pred_flip_' + str(i) + '.jpeg')

# arr = load_data(1)
# resize(1)
# exp, _=load_data_out(15)
# print(exp)

# array = load_loss("/home/venky/time_delay_vp/models/THDloss_model_03_11_2022_15_51/", validation=True)
# print(array.shape)
flip_image()

# test_arr, train_arr, val_arr = loss_model()
# x_test = range(1, len(test_arr)+1)
# x_train = range(len(train_arr))
# x_val = range(len(val_arr))

# plt.plot(x_test, test_arr)
# plt.xlabel("Test cases", fontsize = 25)
# plt.xticks(fontsize = 25)
# plt.ylabel("MAE",fontsize = 25)
# plt.yticks(fontsize = 25)
# plt.title("Test Loss", fontsize = 30)
# plt.show()
