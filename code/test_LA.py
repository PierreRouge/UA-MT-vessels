import os
import argparse
import torch
from networks.unet import TinyUnet
from test_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/', help='Name of Experiment')
parser.add_argument('--model', type=str,  default='UA-MT-2', help='model_name')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--patch_size', nargs='+', type=int, default=[128, 128, 128], help='Patch _size')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "../model/"+FLAGS.model+"/"
test_save_path = "../model/prediction/"+FLAGS.model+"_post/"
if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)

num_classes = 2

with open(FLAGS.root_path + '/../test.list', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + '/' + item.replace('\n', '')+"/mra_norm.h5" for item in image_list]


def test_calculate_metric(epoch_num):
    features = (32, 64, 128, 256)
    kernel_size = (3, 3, 3, 3)
    strides = (1, 2, 2, 2)
    net = TinyUnet(dim=3, in_channel=1, features=features, strides=strides, kernel_size=kernel_size, nclasses=2).cuda() #change to U-Net for fair comparaison
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
    net.load_state_dict(torch.load(save_mode_path))
    print("init weight from {}".format(save_mode_path))
    net.eval()

    avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                               patch_size=tuple(FLAGS.patch_size),
                               save_result=True, test_save_path=test_save_path)

    return avg_metric


if __name__ == '__main__':
    metric = test_calculate_metric(6000)
    print(metric)