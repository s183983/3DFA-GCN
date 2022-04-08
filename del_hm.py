# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 14:10:39 2022

@author: lowes
"""
import os
import glob
import tqdm

path = r"C:\Users\lowes\OneDrive\Skrivebord\DTU\8_semester\Advaced_Geometric_DL\BU_3DFE_3DHeatmaps_crop_2"
path = "/scratch/s183983/data_cropped/" 
vtk_files = glob.glob(os.path.join(path,"**/*.vtk"))


print("\nwriting and cleaning files\n")
for file in tqdm.tqdm(vtk_files):
    new_file = file.replace("_hm", "")
    os.rename(file, new_file)