import itk
import vtk
import numpy as np
import os
import glob
import tqdm
root = "C:/Users/lakri/Desktop/DTU/8.Semester/Special_geo/BU_3DFE_3DHeatmaps_crop_2/"
vtk_files = glob.glob(os.path.join(root,"**/*.vtk"))
from vtk.numpy_interface import dataset_adapter as dsa
#%%
#file = self.file_list[idx]
reader = vtk.vtkPolyDataReader()
reader.SetFileName(vtk_files[0])
reader.Update()
vertices = np.array(reader.GetOutput().GetPoints().GetData())


#lab_name = os.path.join(self.lab_dir,os.path.basename(file).split('.')[0]+".npz")
#loaded = np.load(lab_name)
#TODO
#label_load = loaded["labels"]
#label = label_load.T

poly = np.array(dsa.WrapDataObject(reader.GetOutput()).Polygons)
faces = np.reshape(poly,(-1,4))[:,1:4]
#%%
#mesh = itk.SetPoint(vertices)
meshType3D = itk.Mesh[itk.D, 3, itk.QuadEdgeMeshTraits.D3BBFF]
meshType3D.SetPoint()
#%%
reader = itk.meshread(vtk_files[1])