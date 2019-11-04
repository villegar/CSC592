import glob
import numpy as np
import os
import rasterio as rio
import earthpy as et
import psutil
import random as rand
import ray
import shelve
import skimage.io
import subprocess as shell
import time
#from numpy import zeros
from datetime import datetime
from math import sqrt
from PIL import Image
#for image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Chip_Classify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster)
	tic = time.time()
	Image.show(title=ImageFile)
	#pause(random('beta',1,1)*30)
	#Matlab random('beta',1,1) = Python np.random.beta(1,1)
	time.sleep(np.random.beta(1,1)*30)
	
	ImageIn = mpimg.imread(ImageFile)
	with rio.open(ImageFile) as gtf_img:
		info = gtf_img.meta
	print(time.time()-tic)
	#[ImageRow, ImageColumn, NumberOfBands] = len(ImageIn)
	ImageRow = len(ImageIn[0])
	ImageColumn = len(ImageIn[1])
	NumberOfBands = len(ImageIn[2])
	
	if NumberOfBands > 8
		NumberofBands = NumberofBands - 1
		
	Cluster = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	CountClusterPixels = np.zeros((NumberOfClusters, 1))
	MeanCluster = np.zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResult = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	os.mkdir('local/larry.leigh.temp/')
	print('starting big loop')
	print(time.time()-tic)
	
	for j in range(1, ImageRow)
		#display(num2str(100*j/ImageRow))
		for k in range(1, ImageColumn)
			temp(:) = ImageIn(j, k, 1:NumberOfBands)
			EuclideanDistanceResultant(j, k, :) = sqrt(sum())
		