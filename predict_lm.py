# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 12:33:13 2022

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
from util import main_sample, save_vtk, landmark_regression
from PAConv_model import PAConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_hm(args):
    
    if platform == "win32":
        root = 'C:/Users/lowes/OneDrive/Skrivebord/DTU/8_Semester/Advaced_Geometric_DL/BU_3DFE_3DHeatmaps_crop_2/'
    else:
        if args.full_data:
            root = "/scratch/s183983/BU_3DFE_full/" 
        else:
            root = "/scratch/s183983/data_cropped/" 
        #if opt.user=="s183983" \
            #else "/scratch/s183986/BU_3DFE_3DHeatmaps_crop/"
            
        
    writer = SummaryWriter('runs/3D_face_alignment')
    if args.need_resample:
        main_sample(args.num_points, args.seed, args.sigma, args.sample_way, args.dataset)
    
   
    # data argument
    ScaleAndTranslate = aug.PointcloudScaleAndTranslate()


    # select a model to train
    model = PAConv(args, 83).to(device)   # 68 in FaceScape; 83 in BU-3DFE and FRGC
    model = nn.DataParallel(model)
    names = glob.glob('./checkpoints/%s/models/*.pt' % (args.exp_name))
    names.sort()
    name = names[-1]
    ckpt = torch.load(name, map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt)

    model.eval()

    lm_folder = os.path.join('./checkpoints',args.exp_name,'landmarks')
    if not os.path.exists(lm_folder):
        os.makedirs(lm_folder)
    test_list = glob.glob(os.path.join(root,'test',"*.vtk"))
    lm_l2, hm_l2 = [], []
    for file_id in range(len(test_list)):
        print_set = PrintDataset(root,"test", args.batch_size, args.num_points, use_texture=args.use_texture, file_id=file_id)
        print_loader = DataLoader(print_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
        
        preds = np.full([print_set.batch_size, print_set.mesh_points, 83], np.nan)
        
        for point, landmark, seg, texture, choice in print_loader:
            seg = torch.where(torch.isnan(seg), torch.full_like(seg, 0), seg)
            if args.no_cuda == False:
                point = point.to(device)                   # point: (Batch * num_point * num_dim)
                landmark = landmark.to(device)             # landmark : (Batch * landmark * num_dim)
                seg = seg.to(device)  
                texture = texture.to(device)                     # seg: (Batch * point_num * landmark)
            point_normal = aug.normalize_data(point)           # point_normal : (Batch * num_point * num_dim)
            point_normal = ScaleAndTranslate(point_normal)
            model.eval()
            with torch.no_grad():
                point_normal = point_normal.permute(0, 2, 1)
                pred_heatmap = model(point_normal,texture).cpu()
                
            for i in range(print_loader.batch_size):
                preds[i,choice[i],:] = (pred_heatmap[i,:,:]).T
                
        pred_labels = np.nanmean(preds,axis=0)
        assert not np.isnan(pred_labels).any(), "pred is nan, so nanmean not great"
        lm = landmark_regression(torch.from_numpy(print_set.points), torch.from_numpy(pred_labels), 100)
        np.savetxt(os.path.join(lm_folder,print_set.file_name+"_lm_MDS.txt"),lm.cpu().numpy())
        
        lm_l2.append(torch.sqrt(torch.pow(lm.cpu()-torch.from_numpy(print_set.landmarks),2).sum(0)).mean().numpy())
        hm_l2.append(np.sqrt(np.pow(pred_labels-print_set.label,2).sum(0)).mean())
        
    np.savez(os.path.join(lm_folder, "lm_hm_l2_error.npz"), lm = np.array(lm_l2), hm = np.array(hm_l2))
    for idx in range(pred_labels.shape[1]):
        save_name = os.path.join('./checkpoints',args.exp_name,'meshes',print_set.file_name+'_test_hm'+str(idx+1)+'.vtk')
        save_vtk(print_set.pd,pred_labels[:,idx], save_name)
        
    
if __name__ == "__main__":
        
    args = parser.parse_args()
    init._init_(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    predict_hm(args)