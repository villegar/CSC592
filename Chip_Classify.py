#!/usr/bin/env python3
import numpy as np
import numpy.matlib
import os
import random as rand
import ray
import shelve
import subprocess as shell
import time
#from numpy import zeros
from datetime import datetime
from math import sqrt
from PIL import Image
from utils import save, progbar
from time import sleep

#for image
import matplotlib.pyplot as plt
from skimage.io import imread
import rasterio as rio
#import geopandas as gpd
#import earthpy as et
#import earthpy.spatial as es
#import earthpy.plot as ep

# Start Ray.
#ray.init(num_cpus = 10)

def Chip_Classify0(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
	print("ChipClassify function")
	print(ImageLocation)
	print(SaveLocation)
	print(str(NumberOfClusters))
	print(len(InitialCluster))
	InitialCluster = np.array(InitialCluster).reshape((NumberOfClusters,-1))
	print(str(InitialCluster.shape))

#@ray.remote
#def EuclideanDistance(j, Cluster, CountClusterPixels, EuclideanDistanceResultant, ImageColumn, ImageIn, InitialCluster, MeanCluster, NumberOfBands, NumberOfClusters):
def EuclideanDistance(j, ImageColumn, ImageIn, ImageRow, InitialCluster, NumberOfBands, NumberOfClusters):
	Cluster = np.zeros((1, ImageColumn, NumberOfClusters))
	CountClusterPixels = np.zeros((NumberOfClusters, 1))
	MeanCluster = np.zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResultant = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	for k in range(0, ImageColumn - 1):
		temp = ImageIn[j, k, 0:NumberOfBands]
		#print("Inner loop: ({},{})".format(j,k))
		EuclideanDistanceResultant[j, k, :] = np.sqrt(np.sum(np.power((np.matlib.repmat(temp, NumberOfClusters, 1) - InitialCluster[: ,:]), 2), axis = 1))
		DistanceNearestCluster = min(EuclideanDistanceResultant[j, k, :])

		for l in range(0, NumberOfClusters - 1):
			if DistanceNearestCluster != 0:
				if DistanceNearestCluster == EuclideanDistanceResultant[j, k, l]:
					CountClusterPixels[l] = CountClusterPixels[l] + 1
					for m in range(0, NumberOfBands):
						MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[j, k, m]
					Cluster[0, k, l] = l
	return Cluster

def Chip_Classify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
	tic = time.time()
	sleep(np.random.beta(1,1)*30)
	# Reshape InitialCluster
	InitialCluster = np.array(InitialCluster).reshape((NumberOfClusters,-1))
	ImageIn = imread(ImageFile)
	with rio.open(ImageFile) as gtf_img:
		info = gtf_img.profile
	print(time.time()-tic)
	ImageRow, ImageColumn, NumberOfBands = ImageIn.shape

	if NumberOfBands > 8:
		NumberOfBands = NumberOfBands - 1

	# prealocate
	Cluster = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	CountClusterPixels = np.zeros((NumberOfClusters, 1))
	MeanCluster = np.zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResultant = np.zeros((ImageRow, ImageColumn, NumberOfClusters))
	#os.mkdir('local/larry.leigh.temp/')
	directory = '/tmp/ChipS'
	if not os.path.exists(directory):
		os.makedirs(directory)
	print('starting big loop')
	print(time.time()-tic)

	#Cluster = np.zeros((1, ImageColumn, NumberOfClusters)) # For Ray
	for j in range(0, ImageRow - 1):
		#display(num2str(100*j/ImageRow))
		if(j % 10 == 0):
			progbar(j, ImageRow)
		## The following 6 lines are for Ray (DO NOT DELETE)
		#TaskID = EuclideanDistance.remote(j, ImageColumn, ImageIn, ImageRow, InitialCluster, NumberOfBands, NumberOfClusters)
		#output = ray.get(TaskID)
		#if(output.shape[1:3] == Cluster[1:3]):
		#	Cluster = np.concatenate((Cluster, output))
		#else:
		#	Cluster = np.concatenate((Cluster, np.zeros((1, ImageColumn, NumberOfClusters))))

		for k in range(0, ImageColumn - 1):
			temp = ImageIn[j, k, 0:NumberOfBands]
			EuclideanDistanceResultant[j, k, ] = np.sqrt(np.sum(np.power((np.matlib.repmat(temp, NumberOfClusters, 1) - InitialCluster[: ,:]), 2), axis = 1))
			DistanceNearestCluster = min(EuclideanDistanceResultant[j, k, :])

			for l in range(0, NumberOfClusters - 1):
				if DistanceNearestCluster != 0:
					if DistanceNearestCluster == EuclideanDistanceResultant[j, k, l]:
						CountClusterPixels[l] = CountClusterPixels[l] + 1
						for m in range(0, NumberOfBands - 1):
							MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[j, k, m]
						Cluster[j, k, l] = l
	progbar(ImageRow, ImageRow)
	print('\nfinished big loop')

	ImageDisplay = np.sum(Cluster, axis = 2)
	print(time.time() - tic)

	ClusterPixelCount = np.count_nonzero(Cluster, axis = 2)

	#Calculate TSSE within clusters
	TsseCluster = np.zeros((1, NumberOfClusters))
	CountTemporalUnstablePixel = 0

	for j in range(0, ImageRow - 1):
		for k in range(0, ImageColumn - 1):
			FlagSwitch = int(max(Cluster[j, k, :]))
			print(Cluster[j, k, :])

			#store SSE of related to each pixel
			if FlagSwitch == 0:
				CountTemporalUnstablePixel = CountTemporalUnstablePixel + 1
			else:
				TsseCluster[0,FlagSwitch] = TsseCluster[0,FlagSwitch] + np.sum(np.power( (np.squeeze(ImageIn[j, k, 0:NumberOfBands]) - np.transpose(InitialCluster[FlagSwitch, :])),2), axis = 0)
				#count the number of pixels in each cluster
				#Collected_ClusterPixelCount[FlagSwitch] = Collected_ClusterPixelCount[FlagSwitch] + 1
	Totalsse = np.sum(TsseCluster)
	#get data for final stats....
	#calculate the spatial mean and standard deviation of each cluster

	ClusterMeanAllBands = np.zeros((NumberOfClusters, NumberOfBands))
	ClusterSdAllBands = np.zeros((NUmberOfClusters, NumberOfBands))
	print('finished small loop')
	print(time.time()-tic)

	for i in range(0, NumberOfClusters - 1):
		Temp = Cluster[:, :, i]

		Temp[Temp == i] = 1

		MaskedClusterAllBands = np.apply_along_axes(np.multiply, Temp, ImageIn[:, :, 0:NumberOfBands])

		for j in range(0, NumberOfBands - 1):
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
	save(filename, ['MeanCluster', 'CountClusterPixels', 'ClusterPixelCount', 'ClusterMeanAllBands', 'ClusterSdAllBands', 'Totalsse'])
	print('done!')
	print(time.time()-tic)
