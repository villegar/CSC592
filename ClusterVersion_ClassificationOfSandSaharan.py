# Code By: Mahesh Shrestha
# March 1, 2017

# Data: Google Earth Engine (Landsat 8 OLI)
# Code to perform classification of sand using kmean algorithm
# K mean make the cluster of the similar pixel of the image based on their
# spectral characteristics
# objective: to find the number of cluster in an image having distinct
# spectral characteristics


###########################################################################
###########################################################################
# all pixels of an image are categorised in different clusters based on
# the same spectral characteristics possess by these pixels

# kmean algorithm is used to separate pixels based on their spectral
# characteristics


#  Determining number of clusters
# it is determined by the
# percentage change in TSSE while increasing number of clusters. If
# increasing number of clusters decreases TSSE more than the specified
# threshold then only number of clusters is increased by 1 otherwise
# number of clusters will be same

#  Determining final estimate for each clusters
# for making decision whether new estimate is required or not,
# difference between the present and previous estimate of the mean of
# the clusters are observed. If the observed difference is more than
# the threshold then new mean for the cluster is calculated otherwise
# current mean of cluster is the optimal mean(values having minimum
# variance with all the remaining pixels of the cluster).


###########################################################################
###########################################################################
import glob
import os

ImageLocation       = 'HighRezWorld'
SaveLocation        = 'HighRezFullWorld_100_OutputRun3_Catch'
SummaryLocation     = 'HighRezFullWorld_100_OutputRun3_Catch_Summary'
LogLocation	        = 'logs'
BaseName            = 'HighRezFullWorld_100' ##ok<NOPTS>
override_init       = 1; # set to 1 to start init file below...
override_file       = 'HighRezFullWorld_100_2019.10.30_19.08_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.10.24_09.42_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.10.22_03.12_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.10.18_12.52_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.10.14_13.05_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.09.29_09.36_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.09.12_23.55_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.08.13_08.18_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.08.05_11.00_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.08.01_11.39_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.07.29_15.54_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.07.24_00.41_InitialCluster_C160.mat'
#HighRezFullWorld_100_2019.07.25_17.38_InitialCluster_C160.mat'
# 'HighRezWorld_2019.04.22_14.43_InitialCluster_C19.mat'


directories = glob.glob(ImageLocation + "/*")
print(directories)
#
# for directory = 3:size(directorys,1)
#     ImageList   = dir(fullfile(ImageLocation,directorys(directory).name,'L8*'));
#     ImageLocation_Sub = fullfile(ImageLocation,directorys(directory).name) ##ok<NOPTS>
#     In = dir(fullfile(ImageLocation_Sub,'L*'));
#     display(['current size = ',num2str(size(In,1)),', added to start size of ',num2str(InSize)])
#     InSize = InSize+size(In,1);
#
#     create_job(ImageLocation_Sub,SaveLocation,ImageList,NumberOfClusters,InitialCluster)
#     display([datestr(now),'starting queue'])
#     unix('sbatch job.slurm','-echo');
#     pause(15)
# end
