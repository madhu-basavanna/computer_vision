import os
import random
import argparse
from collections import OrderedDict
from pretrain import MultiCropWrapper
from pretrain import DINOHead
from torch.utils.data import DataLoader

import torch
from pprint import pprint
from torchvision.transforms import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone
from data.pretraining import DataReaderPlainImg



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str,
                        default="/home/madhu/Desktop/dl_lab/cv_assignment/data/crops/images/256")
    parser.add_argument('--weights-init', type=str,
                        default="/home/madhu/Desktop/dl_lab/cv_assignment/data/pretrained_best.pth")
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors",
                     args.weights_init.replace("/", "_").replace("models", "")))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model
    #raise NotImplementedError("TODO: build model and load weights snapshot")
    model = ResNet18Backbone(pretrained=False).cuda()
    #DINO head is not essentail
    model = MultiCropWrapper(model, torch.nn.Identity())

    pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.weights_init)
    dict = torch.load(pretrained_path)['teacher']
    pretrained = OrderedDict()
    #Removing the weights related to Dino head
    for k, v in dict.items():
        if k[:9] == 'backbone.':
            pretrained[k] = v
    model.load_state_dict(pretrained)

    # for nearest neighbors we don't need the last fc layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # dataset
    data_root = args.data_folder
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    val_data = DataReaderPlainImg(os.path.join(data_root, "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=0,
                                             pin_memory=True, drop_last=True)
    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.
    query_indices = [17, 30, 50, 270, 831, 572,725, 325]
    nns = []
    display_image = []

    model = model.cuda()

    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        images = img.cuda(non_blocking=True)
        closest_idx, closest_dist = find_nn(model, images, val_loader, 5)
        #raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")
        print("Closest neighbors for image : ", idx, " -> ", str(closest_idx))
        print("Closest neighbors distance for image : ", idx, " -> ", str(closest_dist))

    for idx, img in enumerate(val_loader):
        if idx in closest_idx:
            display_image.append(img)

    fig = plt.figure(figsize=(32, 32))
    plt.title("comparing nearest neighbour for image 29 (1st image)")
    for i in range(1, len(display_image) + 1):
        img = display_image[i - 1]
        img = torch.squeeze(img).permute(1, 2, 0)
        fig.add_subplot(3, 3, i)
        plt.imshow(img)
    plt.savefig("nearest_neighbour_image_29")
    plt.show()

def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    #raise NotImplementedError("TODO: nearest neighbors retrieval")
    print("inside nn")
    model.eval()
    query_output = model(query_img)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    t = tqdm(loader)
    distances = []
    for images in t:
        images = images.to(device)
        output = model(images)
        d = torch.norm(query_output - output, dim=1, p=None)
        distances.append(d.item())

    distances = np.array(distances)
    closest_idx = np.argsort(distances)[:k]
    closest_dist = distances[closest_idx]
    return closest_idx, closest_dist


if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args) 
