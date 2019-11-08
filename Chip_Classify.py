import glob
import numpy as np
import os
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
import rasterio as rio
import geopanda as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

def Chip_Classify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
	tic = time.time()
	#Image.show(title=ImageFile)
	#pause(random('beta',1,1)*30)
	#Matlab random('beta',1,1) = Python np.random.beta(1,1)
	time.sleep(np.random.beta(1,1)*30)

	ImageIn = mpimg.imread(ImageFile)
	with rio.open(ImageFile) as gtf_img:
		info = gtf_img.profile
	print(time.time()-tic)
	#[ImageRow, ImageColumn, NumberOfBands] = len(ImageIn)
	ImageRow = len(ImageIn)
	ImageColumn = len(ImageIn[0])
	NumberOfBands = len(ImageIn[0][0])

	if NumberOfBands > 8:
		NumberofBands = NumberofBands - 1

	Cluster = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	CountClusterPixels = np.zeros((NumberOfClusters, 1))
	MeanCluster = np.zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResult = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	os.mkdir('local/larry.leigh.temp/')
	print('starting big loop')
	print(time.time()-tic)

	for j in range(1, ImageRow):
		#display(num2str(100*j/ImageRow))
		for k in range(1, ImageColumn):
			temp[:] = ImageIn[j, k, 1:NumberOfBands]
			EuclideanDistanceResultant[j, k, :] = sqrt(sum(np.power((np.matlib.repmat(temp, NumberOfClusters, 1) - InitialCluster[: ,:]), 2), 2))
			DistanceNearestCluster = min(EuclideanDistanceResultant[j, k, :])

			for l in range(1, NumberOfClusters):
				if DistanceNearestCluster != 0:
					if DistanceNearestCluster == EuclideanDistanceResultant[j, k, l]:
						CountClusterPixels[l] = CountClusterPixels[l] + 1
						for m in range(1, NumberOfBands):
							MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[j, k, m]
						Cluster[i, j, k] = l
	print('finished big loop')

	ImageDisplay = np.sum(Cluster, axis = 2)
	print(time.time() - tic)

	ClusterPixelCount = np.count_nonzero(Cluster, axis = 2)

	#Calculate TSSE within clusters
	TsseCluster = np.zeros((1, NumberOfClusters))
	CountTemporalUnstablePixel = 0
	for j in range(1, ImageRow):
		for k in range(1, ImageColumn):
			FlagSwitch = max(Cluster[j, k, :])

			#store SSE of related to each pixel
			if FlagSwitch == 0:
				CountTemporalUnstablePixel = CountTemporalUnstablePixel + 1
			else:
				TsseCluster[FlagSwitch] = TsseCluster[FlagSwitch] + sum(np.power( (np.squeeze(ImageIn[j, k, 1:NumberOfBands]) - InitialCluster[FlagSwitch, :]),2))
				#count the number of pixels in each cluster
				#Collected_ClusterPixelCount[FlagSwitch] = Collected_ClusterPixelCount[FlagSwitch] + 1
	Totalsse = sum(TsseCluster)
	#get data for final stats....
	#calculate the spatial mean and standard deviation of each cluster

	ClusterMeanAllBands = np.zeros((NumberOfClusters, NumberOfBands))
	ClusterSdAllBands = np.zeros((NUmberOfClusters, NumberOfBands))
	print('finished small loop')
	print(time.time()-tic)

	for i in range(1, NumberOfClusters):
		Temp = Cluster[:, :, i]

		Temp[Temp == i] = 1

		MaskedClusterAllBands = np.apply_along_axes(np.multiply, Temp, ImageIn[:, :, 1:NumberOfBands])

		for j in range(1, NumberOfBands):
			#Mean = MaskedClusterAllBands(:,:,j)
			Temp = MaskedClusterAllBands[:, :, j]
			TempNonZero = Temp[Temp != 0]
			TempNonzeronan = TempNonZero[not np.isnan(TempNonZero)]
			#TempNonan = Temp[!np.isnan(Temp)]
			FinalClusterMean[j] = np.mean(tempNonzeronan)
			FinalClusterSd[j] = np.std(tempNonzeronan)

		ClusterMeanAllBands[i, :] = FinalClusterMean[1, :]
		ClusterSdAllBands[i, :] = FinalClusterSd[1, :]

	filename = SaveLocation + 'ImageDisplay_' + ImageFile[len(ImageFile)-32:len(ImageFile)-3] + 'mat'
	save(filename, 'ImageDisplay')

	filename = SaveLocation + 'ClusterCount' + str(NumberOfClusters) + '_' + ImageFile[len(ImageFile)-32:len(ImageFile)-4] + '.tif'

	#geotiffwrite(filename, int8(ImageDisplay), Info.RefMatrix);

	with rio.open(filename, 'w', **info) as dst:
		dst.write(np.int8(ImageDisplay), 1)
	
	filename = SaveLocation + 'Stats_' + ImageFile[len(ImageFile)-32:len(ImageFile)-3] + 'mat'
	save(filename, 'MeanCluster', 'CountClusterPixels', 'ClusterPixelCount', 'ClusterMeanAllBands', 'ClusterSdAllBands', 'Totalsse')
	print('done!')
	print(time.time()-tic)
