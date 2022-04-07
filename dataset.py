import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import vtk


def load_face_data(data):
    Heat_data_sample = np.load('./%s-npy/Heat_data_sample.npy' % data, allow_pickle=True)
    Shape_sample = np.load('./%s-npy/shape_sample.npy' % data, allow_pickle=True)
    landmark_position_select_all = np.load('./%s-npy/landmark_position_select_all.npy' % data, allow_pickle=True)
    if data == 'BU-3DFE' or data == 'FaceScape' or data == 'FRGC':
        return Shape_sample, landmark_position_select_all, Heat_data_sample


class FaceLandmarkData(Dataset):
    def __init__(self, partition='trainval', data='BU-3DFE'):
        if data == 'BU-3DFE' or data == 'FaceScape':
            self.data, self.landmark, self.seg = load_face_data(data)
        if data == 'FRGC':
            self.data, self.landmark, self.seg = load_face_data(data)
        self.partition = partition
        self.DATA = data

    def __getitem__(self, item):
        if self.DATA == 'BU-3DFE' or self.DATA == 'FaceScape':
            data_T, landmark_T, seg_T = torch.Tensor(self.data), torch.Tensor(self.landmark), torch.Tensor(self.seg)
            face = data_T[item]
        if self.DATA == 'FRGC':
            data_T, landmark_T, seg_T = torch.Tensor(self.data), torch.Tensor(self.landmark), torch.Tensor(self.seg)
            face = data_T[item]
        landmark = landmark_T[item]
        heatmap = seg_T[item]
        if self.partition == 'trainval':
            indices = list(range(face.size()[0]))
            np.random.shuffle(indices)
            face = face[indices]
            heatmap = heatmap[indices]
        return face, landmark, heatmap

    def __len__(self):
        return np.array(self.data).shape[0]




class MeshDataset(Dataset):
    def __init__(self, root, mode, use_texture=False):
        self.file_list = glob.glob(os.path.join(root,mode,"*.vtk"))
        self.lab_dir = os.path.join(root,"labels")
        self.land_dir = os.path.join(root,"land_marks")
        self.use_texture = use_texture



    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        vertices = np.array(reader.GetOutput().GetPoints().GetData())

        
        lab_name = os.path.join(self.lab_dir,os.path.basename(file).split('.')[0]+".npz")
        loaded = np.load(lab_name)
        label_load = loaded["label"]
        label = label_load.T
        
        if self.use_texture:
            textures = loaded["texture"]

        landmarks = 0
        
        return vertices, landmarks, label