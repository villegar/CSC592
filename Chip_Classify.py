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
from numpy import linspace
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

CPUS = 20
MEMORY = 20000000000
TASKS_LIMIT = CPUS

# Start Ray.
ray.shutdown()
#ray.init(redis_max_memory=10**11, memory=40000000000, object_store_memory=40000000000)#num_cpus = CPUS, temp_dir = '/tmp')#, memory=40000000000, object_store_memory=40000000000)
#ray.init(num_cpus = CPUS, object_store_memory=MEMORY, memory=CPUS*MEMORY)
ray.init(num_cpus = CPUS)

@ray.remote
def testFun(j,k,l):
	print("Processing j = " + str(j))

@ray.remote
def EuclideanDistance(j, ImageColumn, ImageIn, ImageRow, InitialCluster, NumberOfBands, NumberOfClusters):
	Cluster = zeros((1, ImageColumn, NumberOfClusters))
	CountClusterPixels = zeros((NumberOfClusters, 1))
	MeanCluster = zeros((NumberOfClusters, NumberOfBands))
	EuclideanDistanceResultant = zeros((1, ImageColumn, NumberOfClusters))
	for k in range(0, ImageColumn):
		temp = ImageIn[k, 0:NumberOfBands]
		EuclideanDistanceResultant[0, k, :] = npsqrt(npsum(nppower((matlib.repmat(temp, NumberOfClusters, 1)) - InitialCluster, 2), axis=1))
		DistanceNearestCluster = min(EuclideanDistanceResultant[0, k, :])

		for l in range(0, NumberOfClusters):
			if DistanceNearestCluster != 0:
				if DistanceNearestCluster == EuclideanDistanceResultant[0, k, l]:
					CountClusterPixels[l] = CountClusterPixels[l] + 1
					for m in range(0, NumberOfBands):
						MeanCluster[l, m] = MeanCluster[l, m] + ImageIn[k, m]
					Cluster[0, k, l] = l
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

@ray.remote
def OLDClusterSummary(i, MaskedClusterAllBands, NumberOfBands):
	FinalClusterMean = zeros(NumberOfBands)
	FinalClusterSd = zeros(NumberOfBands)

	for j in range(0, NumberOfBands):
		#Mean = MaskedClusterAllBands(:,:,j)
		Temp = MaskedClusterAllBands[:, :, j]
		TempNonZero = Temp[npnonzero(Temp)]
		TempNonzeronan = TempNonZero[~npisnan(TempNonZero)]
		#TempNonan = Temp[!np.npisnan(Temp)]
		FinalClusterMean[j] = npmean(TempNonzeronan)
		FinalClusterSd[j] = npstd(TempNonzeronan)
	return([FinalClusterMean,FinalClusterSd,i])

@ray.remote
def ClusterSummary(i, j, MaskedCluster):
	# print(str(i) + "," + str(j))
	# print(MaskedCluster.shape)
	Temp = MaskedCluster
	FinalClusterMean = 0
	FinalClusterSd = 0
	NonZeroIndex = npnonzero(Temp)
	if npsum(NonZeroIndex) == 0:
		return([FinalClusterMean,FinalClusterSd,i,j])
	#print(npsum(Temp > 0))
	TempNonZero = Temp[NonZeroIndex]
	#TempNonzeronan = TempNonZero[npisnan(TempNonZero, where=False)]
	TempNonzeronan = TempNonZero[~npisnan(TempNonZero)]
	if(TempNonzeronan.size > 0):
	# FinalClusterMean = npmean(TempNonzeronan)
	# FinalClusterSd = npstd(TempNonzeronan)
		with warnings.catch_warnings():
			warnings.filterwarnings('error')
			try:
				FinalClusterMean = npmean(TempNonZero)
				FinalClusterSd = npstd(TempNonZero)
			except RuntimeWarning:
				FinalClusterMean = 0
				FinalClusterSd = 0
	return([FinalClusterMean,FinalClusterSd,i,j])


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
	CountClusterPixels = zeros((NumberOfClusters, ImageRow))
	MeanCluster = zeros((NumberOfClusters, NumberOfBands, ImageRow))
	EuclideanDistanceResultant = zeros((ImageRow, ImageColumn, NumberOfClusters))

	TaskIDs = list()
	tic = time.time()
	ImageRow = 100
	for j in range(0,ImageRow):
		#display(num2str(100*j/ImageRow))
		if(j % 10 == 0):
			progbar(j, ImageRow)

		#TaskID = testFun.remote(j,j,j)
		TaskID = EuclideanDistance.remote(j, ImageColumn, ImageIn[j,:,:], ImageRow, InitialCluster, NumberOfBands, NumberOfClusters)
		TaskIDs.append(TaskID)
		if(len(TaskIDs) % TASKS_LIMIT == 0):
			ticTasks = time.time()
			Ready,Pending = ray.wait(TaskIDs)
			results = ray.get(Ready)
			for output in results:
				jPrime = output[4]
				Cluster[jPrime,:,:] = output[0]						# Cluster
				CountClusterPixels[:,jPrime] = output[1][:,0]		# CountClusterPixels
				EuclideanDistanceResultant[jPrime,:,:] = output[2]	# EuclideanDistanceResultant
				MeanCluster[:,:,jPrime] = output[3]					# MeanCluster
			TaskIDs = Pending
	results = ray.get(TaskIDs)
	for output in results:
		jPrime = output[4]
		Cluster[jPrime,:,:] = output[0]						# Cluster
		CountClusterPixels[:,jPrime] = output[1][:,0]		# CountClusterPixels
		EuclideanDistanceResultant[jPrime,:,:] = output[2]	# EuclideanDistanceResultant
		MeanCluster[:,:,jPrime] = output[3]					# MeanCluster
	progbar(ImageRow, ImageRow)

	print('\nfinished big loop')
	ImageDisplay = npsum(Cluster, axis = 2)
	print("Execution time: " + str(time.time() - tic))

	# savez("big.loop.parallel",Cluster=Cluster,
	# 				 CountClusterPixels=CountClusterPixels,
	# 				 EuclideanDistanceResultant=EuclideanDistanceResultant,
	# 				 MeanCluster=MeanCluster)

	ClusterPixelCount = count_nonzero(Cluster, axis = 2)
	#print("Non-zero cluster pixels: " + str(ClusterPixelCount))

	#Calculate TSSE within clusters
	TsseCluster = zeros((1, NumberOfClusters))
	CountTemporalUnstablePixel = 0

	# TSSECluster Parallel
	print("Starting TSSE Cluster computation\n")
	tic = time.time()
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
				#TsseCluster = npsum((TsseCluster,output[1]), axis=1)
			TaskIDs = Pending
	results = ray.get(TaskIDs)
	for output in results:
		jPrime = output[2]
		CountTemporalUnstablePixel = CountTemporalUnstablePixel + output[0]
		#TsseCluster = npsum((TsseCluster,output[1]), axis=1)
		TsseCluster = TsseCluster + output[1]
	progbar(ImageRow, ImageRow)
	print('\n')
	Totalsse = npsum(TsseCluster)
	print("Execution time: " + str(time.time() - tic))
	savez("small.loop.parallel",CountTemporalUnstablePixel=CountTemporalUnstablePixel,TsseCluster=TsseCluster)

	#get data for final stats....
	#calculate the spatial mean and standard deviation of each cluster
	ClusterMeanAllBands = zeros((NumberOfClusters, NumberOfBands))
	ClusterSdAllBands = zeros((NumberOfClusters, NumberOfBands))

	# Cluster Summary Parallel
	tic = time.time()
	print("Starting Cluster Summary computation\n")
	TaskIDs = list()
	kValues = linspace(0,ImageColumn,100, dtype=int8)
	for i in range(0, NumberOfClusters):
		Temp = Cluster[:, :, i]
		Temp[Temp == i] = 1
		MaskedClusterAllBands = Temp[:,:,None]*ImageIn[:, :, 0:NumberOfBands]

		if(i % 10 == 0):
			progbar(i, NumberOfClusters)
		for j in range(0, NumberOfBands):
			#for k in range(0, ImageColumn):
			for k in range(1,len(kValues)):
				#TaskID = ClusterSummary.remote(i, j, MaskedClusterAllBands[:,kValues[k-1]:kValues[k],j])
				TaskID = ClusterSummary.remote(i, j, zeros(MaskedClusterAllBands[:,kValues[k-1]:kValues[k],j].shape))
				TaskIDs.append(TaskID)
				#if(len(TaskIDs) % TASKS_LIMIT == 0):
				if(len(TaskIDs) >= TASKS_LIMIT):
					Ready,Pending = ray.wait(TaskIDs)
					results = ray.get(Ready)
					for output in results:
						iPrime = output[2]
						jPrime = output[3]
						ClusterMeanAllBands[iPrime, jPrime] += output[0]
						ClusterSdAllBands[iPrime, jPrime] += output[1]
					TaskIDs = Pending
	results = ray.get(TaskIDs)
	for output in results:
		iPrime = output[2]
		jPrime = output[3]
		FinalClusterMean =  output[0]
		FinalClusterSd = output[1]
		ClusterMeanAllBands[iPrime, jPrime] += output[0]
		ClusterSdAllBands[iPrime, jPrime] += output[1]
	progbar(NumberOfClusters, NumberOfClusters)
	print('\n')

	print("Execution time: " + str(time.time() - tic))
	savez("cluster.summary.parallel",ClusterMeanAllBands=ClusterMeanAllBands,ClusterSdAllBands=ClusterSdAllBands)

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
	print("Overall execution: " + str(time.time()-ticOverall))
