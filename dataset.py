import torch
import numpy as np
from torch.utils.data import Dataset
import os
import glob
import vtk
from vtk.numpy_interface import dataset_adapter as dsa



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
    def __init__(self, root, mode, num_points=3346, use_texture=False):
        self.file_list = glob.glob(os.path.join(root,mode,"*.vtk"))
        self.lab_dir = os.path.join(root,"labels")
        self.land_dir = os.path.join(root,"land_marks")
        self.use_texture = use_texture
        self.num_points = num_points

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
        #TODO
        label_load = loaded["labels"]
        label = label_load.T
        
        

        landmarks = np.zeros((84,3))
        
        choice = np.random.choice(len(vertices), self.num_points, replace=False)

        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        
        if self.use_texture:
            textures = loaded["texture"]
            textures = textures[choice].astype(np.float32)
            textures = textures/255*2-1
        else:
            textures = np.empty(0)
        
        vertices = vertices[choice, :]
        label = label[choice]

        return torch.from_numpy(vertices), torch.from_numpy(landmarks), torch.from_numpy(label), torch.from_numpy(textures)
    
    
    
class PrintDataset(Dataset):
    def __init__(self, root, mode, batch_size, num_points=3346, use_texture=False, file_id = 0, no_gt = False):
        files = glob.glob(os.path.join(root,mode,"*.vtk"))
        files.sort()
        self.file_list = [files[file_id] for _ in range(batch_size)]
        self.file_name = os.path.basename(files[file_id]).split('.')[0]
        self.lab_dir = os.path.join(root,"labels")
        self.use_texture = use_texture
        self.num_points = num_points
        self.batch_size = batch_size
        file = files[file_id]
        
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        vertices = np.array(reader.GetOutput().GetPoints().GetData())
        
        lab_name = os.path.join(self.lab_dir,os.path.basename(file).split('.')[0]+".npz")
        loaded = np.load(lab_name)
        #TODO

        
        self.mesh_points = len(vertices)
        
        self.sample_size = np.floor(self.mesh_points/batch_size).astype(np.int32)
        self.indices = np.arange(self.mesh_points)
        np.random.shuffle(self.indices)
        poly = np.array(dsa.WrapDataObject(reader.GetOutput()).Polygons)
        self.points = vertices
        self.faces = np.reshape(poly,(-1,4))[:,1:4]
        self.pd = reader.GetOutput()
        if no_gt:
            self.no_gt = False
        else:
            
            label_load = loaded["labels"]
            self.label = label_load.T
            self.landmarks = loaded["landmarks"][:self.label.shape[1]]
        
        if self.use_texture:
            self.textures = loaded["texture"]


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        
        
        sample = self.indices[idx*self.sample_size:] if idx==self.batch_size\
            else self.indices[idx*self.sample_size:(idx+1)*self.sample_size]
        
        choices = np.arange(self.mesh_points)
        
        choices = choices[np.isin(choices,sample,invert=True)]
        
        choice = np.random.choice(choices, self.num_points-len(sample), replace=False)
        choice = np.concatenate((choice,sample), axis=0)
        # resample
        # note that the number of points in some points clouds is less than 2048, thus use random.choice
        # remember to use the same seed during train and test for a getting stable result
        
        if self.use_texture:
            textures = self.textures[choice].astype(np.float32)
            textures = textures/255*2-1
        else:
            textures = np.empty(0)
            
        vertices = self.points[choice, :]
        if self.no_gt:
            return torch.from_numpy(vertices), torch.from_numpy(textures), choice
        else:
            label = self.label[choice]
            
            return torch.from_numpy(vertices), torch.from_numpy(self.landmarks), torch.from_numpy(label),\
                torch.from_numpy(textures), choice