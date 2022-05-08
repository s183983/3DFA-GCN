# -*- coding: utf-8 -*-
"""
Created on Sun May  8 15:11:12 2022

@author: lowes
"""

import os
import math
import time
import numpy as np
from scipy import io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sys import platform
import wandb
import glob
import init
from My_args import parser
import augmentations as aug
from dataset import MeshDataset, PrintDataset
from loss import AdaptiveWingLoss
from util import lm_weighted_avg, save_vtk, landmark_regression
from PAConv_model import PAConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_lm_no_gt(args):
    
    if platform == "win32":
        root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop_2/'
    else:
        if args.full_data:
            root = "/scratch/s183983/MBJ_full/" 
        else:
            root = "/scratch/s183983/data_cropped/" 
        #if opt.user=="s183983" \
            #else "/scratch/s183986/BU_3DFE_3DHeatmaps_crop/"
            
        
    
   
    # data argument
    ScaleAndTranslate = aug.PointcloudScaleAndTranslate()

    save_hm_list = ["20220421135546",
                    "20220421135108",
                    "20220421134808",
                    "20220421135242"]
    # select a model to train
    model = PAConv(args, 84).to(device)   # 68 in FaceScape; 84 in BU-3DFE and FRGC
    model = nn.DataParallel(model)
    names = glob.glob('./checkpoints/%s/models/*.pt' % (args.exp_name))
    names.sort()
    name = names[-1]
    ckpt = torch.load(name, map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt)

    model.eval()
    mesh_folder = os.path.join('./checkpoints',args.exp_name,'meshes_MBJ')
    if not os.path.exists(mesh_folder):
        os.makedirs(mesh_folder)
    lm_folder = os.path.join('./checkpoints',args.exp_name,'landmarks_MBJ')
    if not os.path.exists(lm_folder):
        os.makedirs(lm_folder)
    test_list = glob.glob(os.path.join(root,'meshes',"*.vtk"))
    model.eval()

    for file_id in range(len(test_list)):
        print_set = PrintDataset(root,"meshes", args.batch_size,
                                 num_points=args.num_points,
                                 use_texture=args.use_texture,
                                 file_id=file_id,
                                 no_gt = True)
        print_loader = DataLoader(print_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        preds = np.full([2*print_set.batch_size, print_set.mesh_points, 84], np.nan)
        print("predicting lm for", print_set.file_name)
        for point, texture, choice in print_loader:
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                texture = texture.to(device)                     # seg: (Batch * point_num * landmark)
            point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            with torch.no_grad():
                point_normal = point_normal.permute(0, 2, 1)
                pred_heatmap = model(point_normal,texture).cpu()
                
            for i in range(print_loader.batch_size):
                preds[i,choice[i],:] = (pred_heatmap[i,:,:]).T
        for point, texture, choice in print_loader:
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                texture = texture.to(device)                     # seg: (Batch * point_num * landmark)
            point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            with torch.no_grad():
                point_normal = point_normal.permute(0, 2, 1)
                pred_heatmap = model(point_normal,texture).cpu()
                
            for i in range(print_loader.batch_size):
                preds[print_set.batch_size+i,choice[i],:] = (pred_heatmap[i,:,:]).T
        pred_labels = np.nanmean(preds,axis=0)
        
        pred_labels = np.nan_to_num(pred_labels)
        
        if np.isnan(pred_labels).any():
            print("pred is nan, :(")
            continue
        
        lm = landmark_regression(torch.from_numpy(print_set.points), torch.from_numpy(pred_labels), 100)
        lm_avg = lm_weighted_avg(print_set.pd, pred_labels)
        np.savetxt(os.path.join(lm_folder,print_set.file_name+"_lm_MDS.txt"),lm.cpu().numpy())
        np.savetxt(os.path.join(lm_folder,print_set.file_name+"_lm_AVG.txt"),lm_avg)
                
        if print_set.file_name in save_hm_list:
            for idx in range(pred_labels.shape[1]):
                save_name = os.path.join(mesh_folder,print_set.file_name+'_hm_'+str(idx+1)+'.vtk')
                save_vtk(print_set.pd,pred_labels[:,idx], save_name)
        
    
if __name__ == "__main__":
        
    args = parser.parse_args()
    init._init_(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    predict_lm_no_gt(args)