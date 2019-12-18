#!/usr/bin/env python3
#import numpy as np

#-------------------
from numpy import zeros
from numpy import sqrt as npsqrt
from numpy import array
from numpy import sum as npsum
from numpy import matlib
from numpy import power as nppower
from numpy import count_nonzero
from numpy import squeeze
from numpy import transpose
from numpy import apply_along_axis
from numpy import multiply
from numpy import isnan as npisnan
from numpy import nanmean as npmean
from numpy import nanstd as npstd
from numpy import int8
from numpy import random
from numpy import nonzero as npnonzero
from numpy import save
from numpy import savez
from numpy import concatenate
from numpy import dstack
from bson import ObjectId
#-------------------

import warnings
import numpy.matlib
import os
import random as rand
import ray
import shelve
import subprocess as shell
import time
#from numpy import zeros
from datetime import datetime
#from math import sqrt
from PIL import Image
from utils import progbar
from utils import save as shelver
from time import sleep

#for image
import matplotlib.pyplot as plt
from skimage.io import imread
import rasterio as rio
#import geopandas as gpd
#import earthpy as et
#import earthpy.spatial as es
#import earthpy.plot as ep
#import numpy as np

def Chip_Classify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
	ticOverall = time.time()
	#sleep(random.beta(1,1)*30)
	# Reshape InitialCluster
	InitialCluster = array(InitialCluster).reshape((NumberOfClusters,-1))
	ImageIn = imread(ImageFile)
	with rio.open(ImageFile) as gtf_img:
		Info = gtf_img.profile
		Info.update(dtype=rio.int8)
	#print(time.time()-tic)
	ImageRow, ImageColumn, NumberOfBands = ImageIn.shape
	if NumberOfBands > 8:
		NumberOfBands = NumberOfBands - 1
	# prealocate
	Cluster = zeros((ImageRow, ImageColumn, NumberOfClusters))
	CountClusterPixels = zeros((NumberOfClusters, 1))
	MeanCluster = zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResultant = zeros((ImageRow, ImageColumn, NumberOfClusters))
	#os.mkdir('local/larry.leigh.temp/')
	directory = '/tmp/ChipS'
	if not os.path.exists(directory):
		os.makedirs(directory)
	print('starting big loop')
	tic = time.time()
	for j in range(0,ImageRow):
		# if(j % 10 == 0):
			# progbar(j, ImageRow)

		for k in range(0, ImageColumn):
			temp = ImageIn[j, k, 0:NumberOfBands]

			#EuclideanDistanceResultant[j, k, :] = np.npsqrt(np.npsum(np.nppower(np.subtract(np.matlib.repmat(temp, NumberOfClusters, 1), InitialCluster[: ,:]), 2), axis = 1))
			EuclideanDistanceResultant[j, k, :] = npsqrt(npsum(nppower((matlib.repmat(temp, NumberOfClusters, 1)) - InitialCluster, 2), axis=1))
			DistanceNearestCluster = min(EuclideanDistanceResultant[j, k, :])

			#print(str(j) +" "+ str(k))

			for l in range(0, NumberOfClusters):
				if DistanceNearestCluster != 0:
					if DistanceNearestCluster == EuclideanDistanceResultant[j, k, l]:
						CountClusterPixels[l] = CountClusterPixels[l] + 1
						for m in range(0, NumberOfBands):
							MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[j, k, m]
						Cluster[j, k, l] = l

	# progbar(ImageRow, ImageRow)

	print('\n')
	# print(Cluster.shape)
	# print(CountClusterPixels.shape)
	# print(EuclideanDistanceResultant.shape)
	# print(MeanCluster.shape)
	print('\nfinished big loop')
	ImageDisplay = npsum(Cluster, axis = 2)
	print("Execution time: " + str(time.time() - tic))
	#print(globals())
	#shelver("big.loop",['Cluster','CountClusterPixels','EuclideanDistanceResultant','MeanCluster'])
	savez("big.loop.serial",Cluster=Cluster,
					 CountClusterPixels=CountClusterPixels,
					 EuclideanDistanceResultant=EuclideanDistanceResultant,
					 MeanCluster=MeanCluster)

	ClusterPixelCount = count_nonzero(Cluster, axis = 2)
	print("Non-zero cluster pixels: " + str(ClusterPixelCount))

	#Calculate TSSE within clusters
	TsseCluster = zeros((1, NumberOfClusters))
	CountTemporalUnstablePixel = 0
	# TSSECluster Serial
	print("Starting TSSE Cluster computation (Serial version)\n")
	tic = time.time()
	for j in range(0, ImageRow):
		for k in range(0, ImageColumn):
			FlagSwitch = int(max(Cluster[j, k, :]))
			#print(Cluster[j, k, :]) #This prints to the log

			#store SSE of related to each pixel
			if FlagSwitch == 0:
				CountTemporalUnstablePixel = CountTemporalUnstablePixel + 1
			else:
				#Might be TsseCluster[0,FlagSwitch-1]
				#TsseCluster[0,FlagSwitch - 1] = TsseCluster[0,FlagSwitch - 1] + np.sum(np.power(np.subtract(np.squeeze(ImageIn[j, k, 0:NumberOfBands - 1]), np.transpose(InitialCluster[FlagSwitch - 1, :])),2), axis = 0)

				TsseCluster[0,FlagSwitch] = TsseCluster[0,FlagSwitch] + npsum(nppower((squeeze(ImageIn[j, k, 0:NumberOfBands]) - transpose(InitialCluster[FlagSwitch, :])),2))

				#count the number of pixels in each cluster
				#Collected_ClusterPixelCount[FlagSwitch] = Collected_ClusterPixelCount[FlagSwitch] + 1
	Totalsse = npsum(TsseCluster)
	print("Execution time: " + str(time.time() - tic))
	savez("small.loop.serial",CountTemporalUnstablePixel=CountTemporalUnstablePixel,TsseCluster=TsseCluster)

	#get data for final stats....
	#calculate the spatial mean and standard deviation of each cluster

	ClusterMeanAllBands = zeros((NumberOfClusters, NumberOfBands))
	ClusterSdAllBands = zeros((NumberOfClusters, NumberOfBands))
	print('finished small loop')
	#print(time.time()-tic)

	# Cluster Summary Serial
	tic = time.time()
	FinalClusterMean = zeros(NumberOfBands)
	FinalClusterSd = zeros(NumberOfBands)

	for i in range(0, NumberOfClusters):
		Temp = Cluster[:, :, i]

		Temp[Temp == i] = 1

		MaskedClusterAllBands = Temp[:,:,None]*ImageIn[:, :, 0:NumberOfBands]

		for j in range(0, NumberOfBands):
			#Mean = MaskedClusterAllBands(:,:,j)
			Temp = MaskedClusterAllBands[:, :, j]
			TempNonZero = Temp[npnonzero(Temp)]
			TempNonzeronan = TempNonZero[~npisnan(TempNonZero)]
			#TempNonan = Temp[!np.isnan(Temp)]
			with warnings.catch_warnings():
				warnings.filterwarnings('error')
				try:
					FinalClusterMean[j] = npmean(TempNonZero)
					FinalClusterSd[j] = npstd(TempNonZero)
				except RuntimeWarning:
					FinalClusterMean[j] = 0
					FinalClusterSd[j] = 0

		ClusterMeanAllBands[i, :] = FinalClusterMean[:]
		ClusterSdAllBands[i, :] = FinalClusterSd[:]

	print("Execution time: " + str(time.time() - tic))
	savez("cluster.summary.serial",ClusterMeanAllBands=ClusterMeanAllBands,ClusterSdAllBands=ClusterSdAllBands)
	filename = str(SaveLocation) + 'ImageDisplay_' + ImageFile[len(ImageFile)-32:len(ImageFile)-3] + 'mat'
	print('Got filename. Now save the data')
	print(filename)
	save(filename, ImageDisplay)

	filename = str(SaveLocation) + 'ClusterCount' + str(NumberOfClusters) + '_' + ImageFile[len(ImageFile)-32:len(ImageFile)-4] + '.tif'

	#geotiffwrite(filename, int8(ImageDisplay), Info.RefMatrix);

	with rio.open(filename, 'w', **Info) as dst:
		dst.write(int8(ImageDisplay), 1)

	filename = str(SaveLocation) + 'Stats_' + ImageFile[len(ImageFile)-32:len(ImageFile)-3] + 'mat'
	savez(filename, [MeanCluster, CountClusterPixels, ClusterPixelCount, ClusterMeanAllBands, ClusterSdAllBands, Totalsse])
	print('done!')
	print(time.time()-ticOverall)
