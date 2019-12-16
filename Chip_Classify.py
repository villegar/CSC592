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
from numpy import isnan
from numpy import mean
from numpy import std
from numpy import int8
from numpy import random
from numpy import nonzero
from numpy import save
from numpy import savez
from numpy import concatenate
from numpy import dstack
from bson import ObjectId
#-------------------


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

CPUS = 20
TASKS_LIMIT = CPUS

# Start Ray.
ray.shutdown()
#ray.init(redis_max_memory=10**11, memory=40000000000, object_store_memory=40000000000)#num_cpus = CPUS, temp_dir = '/tmp')#, memory=40000000000, object_store_memory=40000000000)
ray.init(num_cpus = CPUS)

def Chip_Classify0(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
	print("ChipClassify function")
	print(ImageLocation)
	print(SaveLocation)
	print(str(NumberOfClusters))
	print(len(InitialCluster))
	InitialCluster = array(InitialCluster).reshape((NumberOfClusters,-1))
	print(str(InitialCluster.shape))


#def EuclideanDistance(j, Cluster, CountClusterPixels, EuclideanDistanceResultant, ImageColumn, ImageIn, InitialCluster, MeanCluster, NumberOfBands, NumberOfClusters):
@ray.remote
def testFun(j,k,l):
	print("Processing j = " + str(j))

@ray.remote
def EuclideanDistance(j, ImageColumn, ImageIn, ImageRow, InitialCluster, NumberOfBands, NumberOfClusters):
	Cluster = zeros((1, ImageColumn, NumberOfClusters))
	CountClusterPixels = zeros((NumberOfClusters, 1))
	MeanCluster = zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResultant = zeros((1, ImageColumn, NumberOfClusters))
	#print(ImageIn.shape)
	for k in range(0, ImageColumn):
		temp = ImageIn[k, 0:NumberOfBands]
		# t1 = (matlib.repmat(temp, NumberOfClusters, 1))
		# t2 = nppower(t1 - InitialCluster, 2)
		# EuclideanDistanceResultant[0, k, :] = npsqrt(npsum(t2, axis=1))
		EuclideanDistanceResultant[0, k, :] = npsqrt(npsum(nppower((matlib.repmat(temp, NumberOfClusters, 1)) - InitialCluster, 2), axis=1))
		DistanceNearestCluster = min(EuclideanDistanceResultant[0, k, :])

		for l in range(0, NumberOfClusters):
			if DistanceNearestCluster != 0:
				if DistanceNearestCluster == EuclideanDistanceResultant[0, k, l]:
					CountClusterPixels[l] = CountClusterPixels[l] + 1
					for m in range(0, NumberOfBands):
						MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[k, m]
					Cluster[0, k, l] = l
	#return([j])
	return([Cluster,CountClusterPixels,EuclideanDistanceResultant,MeanCluster,j])

@ray.remote
def TSSECluster(j, Cluster, ImageColumn, ImageIn, InitialCluster, NumberOfBands, NumberOfClusters):
	CountTemporalUnstablePixel = 0
	TsseCluster = zeros((1, NumberOfClusters))
	for k in range(0, ImageColumn):
		FlagSwitch = int(max(Cluster[k, :]))

		#store SSE of related to each pixel
		if FlagSwitch == 0:
			CountTemporalUnstablePixel = CountTemporalUnstablePixel + 1
		else:
			TsseCluster[0,FlagSwitch] = TsseCluster[0,FlagSwitch] + npsum(nppower((squeeze(ImageIn[k, 0:NumberOfBands]) - transpose(InitialCluster[FlagSwitch, :])),2))
	return([CountTemporalUnstablePixel,TsseCluster,j])

def Chip_Classify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
	tic = time.time()
	#sleep(random.beta(1,1)*30)
	# Reshape InitialCluster
	InitialCluster = array(InitialCluster).reshape((NumberOfClusters,-1))
	ImageIn = imread(ImageFile)
	with rio.open(ImageFile) as gtf_img:
		Info = gtf_img.profile
		Info.update(dtype=rio.int8)
	print(time.time()-tic)
	ImageRow, ImageColumn, NumberOfBands = ImageIn.shape
	if NumberOfBands > 8:
		NumberOfBands = NumberOfBands - 1
	# prealocate
	#Cluster = zeros((ImageRow, ImageColumn, NumberOfClusters))
	#CountClusterPixels = zeros((NumberOfClusters, 1))
	#MeanCluster = zeros((NumberOfClusters, NumberOfBands))
	#EuclideanDistanceResultant = zeros((ImageRow, ImageColumn, NumberOfClusters))
	#os.mkdir('local/larry.leigh.temp/')
	directory = '/tmp/ChipS'
	if not os.path.exists(directory):
		os.makedirs(directory)
	print('starting big loop')
	print(time.time()-tic)

	#Cluster = zeros((1, ImageColumn, NumberOfClusters)) # For Ray
	# Cluster = zeros((1, 1, 1)) # For Ray
	# CountClusterPixels = zeros((1, 1))
	# MeanCluster = zeros((1, 1))
	# EuclideanDistanceResultant = zeros((1, 1, 1))
	Cluster = zeros((ImageRow, ImageColumn, NumberOfClusters))
	CountClusterPixels = zeros((NumberOfClusters, ImageRow))
	MeanCluster = zeros((NumberOfClusters, NumberOfBands, ImageRow))
	EuclideanDistanceResultant = zeros((ImageRow, ImageColumn, NumberOfClusters))
	#ImageRow = 100
	TaskIDs = list()
	ReadyIDs = list()
	for j in range(0,ImageRow):
		#display(num2str(100*j/ImageRow))
		if(j % 10 == 0):
			progbar(j, ImageRow)
		# The following 6 lines are for Ray (DO NOT DELETE)
		#TaskID = testFun.remote(j,j,j)
		TaskID = EuclideanDistance.remote(j, ImageColumn, ImageIn[j,:,:], ImageRow, InitialCluster, NumberOfBands, NumberOfClusters)
		TaskIDs.append(TaskID)
		#output = ray.get(TaskID)
		if(len(TaskIDs) % TASKS_LIMIT == 0):
		#if(len(TaskIDs) == CPUS): # Retrieve the results after [CPUS] jobs have been spawned
			#print("Processing tasks: ")
			#print(len(TaskIDs))
			ticTasks = time.time()
			Ready,Pending = ray.wait(TaskIDs)
			#ReadyIDs.append(Ready)
			results = ray.get(Ready)
			# if(len(ReadyIDs) % TASKS_LIMIT == 0):
			# 	results = ray.get(ReadyIDs)
			# 	#results = ray.get(TaskIDs)
			# 	#print(time.time() - tic)
			# 	#print(len(Pending))
			# 	print("Processing: " + str(len(Ready)) + "-" + str(len(Pending)))
			for output in results:
				jPrime = output[4]
				Cluster[jPrime,:,:] = output[0]						# Cluster
				CountClusterPixels[:,jPrime] = output[1][:,0]			# CountClusterPixels
				EuclideanDistanceResultant[jPrime,:,:] = output[2]	# EuclideanDistanceResultant
				MeanCluster[:,:,jPrime] = output[3]					# MeanCluster
				# if((Cluster.shape == array([1,1,1])).all()):
				# 	Cluster = output[0] 					# Cluster
				# 	CountClusterPixels = output[1] 			# CountClusterPixels
				# 	EuclideanDistanceResultant = output[2]	# EuclideanDistanceResultant
				# 	MeanCluster = output[3]					# MeanCluster
				# else:
				# 	Cluster = concatenate((Cluster, output[0]))
				# 	CountClusterPixels = concatenate((CountClusterPixels, output[1]), axis = 1)
				# 	EuclideanDistanceResultant = concatenate((EuclideanDistanceResultant, output[2]))
				# 	MeanCluster = dstack((MeanCluster, output[3]))
			# 	# #TaskIDs = list()
			# 	ReadyIDs = list()
			# else:
			# 	print("Tasks")
			# 	print(TaskIDs)
			# 	print("Ready")
			# 	print(ReadyIDs)
			# 	print("Pending")
			# 	print(Pending)
			# 	TaskIDs.remove(ObjectId(ReadyIDs))
			TaskIDs = Pending
			#print(time.time()-ticTasks)

		# if((output.shape[1:3] == Cluster.shape[1:3])):
		# 	Cluster = concatenate((Cluster, output))
		# else:
		# 	Cluster = output
			#Cluster = concatenate((Cluster, zeros((1, ImageColumn, NumberOfClusters))))

		# for k in range(0, ImageColumn):
		# 	temp = ImageIn[j, k, 0:NumberOfBands]
		#
		# 	#EuclideanDistanceResultant[j, k, :] = np.npsqrt(np.npsum(np.nppower(np.subtract(np.matlib.repmat(temp, NumberOfClusters, 1), InitialCluster[: ,:]), 2), axis = 1))
		# 	EuclideanDistanceResultant[j, k, :] = npsqrt(npsum(power((matlib.repmat(temp, NumberOfClusters, 1) - InitialCluster[:, :]), 2), axis=1))
		# 	DistanceNearestCluster = min(EuclideanDistanceResultant[j, k, :])
		#
		# 	#print(str(j) +" "+ str(k))
		#
		# 	for l in range(0, NumberOfClusters):
		# 		if DistanceNearestCluster != 0:
		# 			if DistanceNearestCluster == EuclideanDistanceResultant[j, k, l]:
		# 				CountClusterPixels[l] = CountClusterPixels[l] + 1
		# 				for m in range(0, NumberOfBands):
		# 					MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[j, k, m]
		# 				Cluster[j, k, l] = l
	results = ray.get(TaskIDs)
	for output in results:
		jPrime = output[4]
		Cluster[jPrime,:,:] = output[0]						# Cluster
		CountClusterPixels[:,jPrime] = output[1][:,0]		# CountClusterPixels
		EuclideanDistanceResultant[jPrime,:,:] = output[2]	# EuclideanDistanceResultant
		MeanCluster[:,:,jPrime] = output[3]					# MeanCluster
	progbar(ImageRow, ImageRow)

	print('\n')
	print(Cluster.shape)
	print(CountClusterPixels.shape)
	print(EuclideanDistanceResultant.shape)
	print(MeanCluster.shape)
	print('\nfinished big loop')

	#shelver("big.loop",['Cluster','CountClusterPixels','EuclideanDistanceResultant','MeanCluster'])

	ImageDisplay = npsum(Cluster, axis = 2)
	print(time.time() - tic)

	ClusterPixelCount = count_nonzero(Cluster, axis = 2)
	print("Non-zero cluster pixels: " + str(ClusterPixelCount))

	#Calculate TSSE within clusters
	TsseCluster = zeros((1, NumberOfClusters))
	CountTemporalUnstablePixel = 0

	# TSSECluster Parallel
	print("Starting TSSE Cluster computation\n")
	TaskIDs = list()
	for j in range(0, ImageRow):
		if(j % 10 == 0):
			progbar(j, ImageRow)
		TaskID = TSSECluster.remote(j, Cluster[j,:,:], ImageColumn, ImageIn[j,:,:], InitialCluster, NumberOfBands, NumberOfClusters)
		TaskIDs.append(TaskID)
		if(len(TaskIDs) % TASKS_LIMIT == 0):
			Ready,Pending = ray.wait(TaskIDs)
			results = ray.get(Ready)
			for output in results:
				jPrime = output[2]
				CountTemporalUnstablePixel = CountTemporalUnstablePixel + output[0]
				TsseCluster = TsseCluster + output[1]
			TaskIDs = Pending
	results = ray.get(Ready)
	for output in results:
		jPrime = output[2]
		CountTemporalUnstablePixel = CountTemporalUnstablePixel + output[0]
		TsseCluster = TsseCluster + output[1]
	progbar(ImageRow, ImageRow)
	print('\n')
	Totalsse = npsum(TsseCluster)

	savez("small.loop.parallel",[CountTemporalUnstablePixel,TsseCluster])
	print("Unstable Pixels: " + str(CountTemporalUnstablePixel))
	print("Total SSE: " + str(Totalsse))
	print(TsseCluster[0,1])

	#Calculate TSSE within clusters
	TsseCluster = zeros((1, NumberOfClusters))
	CountTemporalUnstablePixel = 0
	# TSSECluster Serial
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
	savez("small.loop.serial",[CountTemporalUnstablePixel,TsseCluster])
	print("Unstable Pixels: " + str(CountTemporalUnstablePixel))
	print("Total SSE: " + str(Totalsse))
	print(TsseCluster[0,1])

	#get data for final stats....
	#calculate the spatial mean and standard deviation of each cluster

	ClusterMeanAllBands = zeros((NumberOfClusters, NumberOfBands))
	ClusterSdAllBands = zeros((NumberOfClusters, NumberOfBands))
	print('finished small loop')
	print(time.time()-tic)

	FinalClusterMean = zeros(NumberOfBands)
	FinalClusterSd = zeros(NumberOfBands)

	for i in range(0, NumberOfClusters):
		Temp = Cluster[:, :, i]

		Temp[Temp == i] = 1

		MaskedClusterAllBands = Temp[:,:,None]*ImageIn[:, :, 0:NumberOfBands]

		for j in range(0, NumberOfBands):
			#Mean = MaskedClusterAllBands(:,:,j)
			Temp = MaskedClusterAllBands[:, :, j]
			TempNonZero = Temp[nonzero(Temp)]
			TempNonzeronan = TempNonZero[~isnan(TempNonZero)]
			#TempNonan = Temp[!np.isnan(Temp)]
			FinalClusterMean[j] = mean(TempNonzeronan)
			FinalClusterSd[j] = std(TempNonzeronan)

		ClusterMeanAllBands[i, :] = FinalClusterMean[:]
		ClusterSdAllBands[i, :] = FinalClusterSd[:]

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
	print(time.time()-tic)
