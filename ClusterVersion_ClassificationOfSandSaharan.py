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
import numpy as np
import os
import psutil
import random as rand
import ray
import shelve
import skimage.io
import subprocess as shell
import time
#from numpy import zeros
from datetime import datetime
#from math import sqrt
from PIL import Image
from time import sleep

ImageLocation       = 'HighRezWorld'
SaveLocation        = 'HighRezFullWorld_100_OutputRun3_Catch'
SummaryLocation     = 'HighRezFullWorld_100_OutputRun3_Catch_Summary'
LogLocation	        = 'logs'
BaseName            = 'HighRezFullWorld_100'
override_init       = 0 # set to 1 to start init file below...
override_file       = 'InitialConditions.mat.dat'

directories         = glob.glob(ImageLocation + '/Other/L8_BA_R030_V4_Lat0034_Lon0024.tif') #was + '/*')

#ImageList           = glob.glob(ImageLocation + '/NorthAfrica/L8*')
ImageList           = [os.path.basename(x) for x in glob.glob(ImageLocation + '/Other/L8_BA_R030_V4_Lat0034_Lon0024.tif')] #was + '/Other/L8*')]
print("we are in the main file\n\n\n\n\n\n\n\n\n\n");
#start with the number of clusters
NumberOfClusters    = 160 #was 160
NumberOfBands       = 16 #was 16

#flag to check whether to increase clusters or not
FlagCluster         = 1
ThresholdIteration  = 0.0001
CountWhile          = 0

#local loops in serial, set to 0 to run cluster jobs
local_version       = 0

InitialCluster      = np.empty((0,NumberOfBands), dtype=np.float32)
username            = 'Roberto.VillegasDiaz'

numCPUs = psutil.cpu_count(logical=False)
#ray.init(num_cpus=numCPUs)
print(numCPUs)
# L8_BA_R030_V4_Lat0030_Lon0006.tif (270)[793]

from Chip_Classify import Chip_Classify as chipClassify
#@ray.remote
#def chipClassify(ImageLocation_Sub,SaveLocation,ChipPath,NumberOfClusters,InitialCluster):
#    print('Classifying chip')
    #sleep(1)

from createJob import createJob
#def createJob(ImageLocation_Sub,SaveLocation,ImageList,NumberOfClusters,InitialCluster,fileORlist):
#    print('Creating job')
#    #sleep(1)

def dateNow(format = None):
    if(format != None):
        return(datetime.now().strftime(format))
    return(str(datetime.now()))

def findNonCompletedAndResubmit(Jobs,Out,InSize,SaveLocation,NumberOfClusters,InitialCluster):
    print('Finding non-completed jobs and resubmitting')
    #sleep(1)

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[0])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def load(filename):
    existingShelf = shelve.open(filename)
    for key in existingShelf:
        globals()[key]=existingShelf[key]
    existingShelf.close()
    # Reference: https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

def delete(directory, pattern):
    # Get a list of all the file paths that math pattern inside directory
    fileList = glob.glob(directory + '/' + pattern)

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print('Error while deleting file : ' + filePath)
    # Reference: https://thispointer.com/python-how-to-remove-files-by-matching-pattern-wildcards-certain-extensions-only/

def save(filename, variables):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory) and directory != '':
        os.makedirs(directory)
    newShelf = shelve.open(filename,'n') # 'n' for new
    if(type(variables) != type(list())):
        variables = [variables]
    for key in variables:
        try:
            newShelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, newShelf, and imported modules can not be shelved.
            #
            print('ERROR shelving: {0}'.format(key))
    newShelf.close()
    # Reference: https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

def sbatch(jobFile, dependency):
    if(dependency == ''):
        proc = shell.Popen(['sbatch ' + jobFile], shell = True, stdout = shell.PIPE)
    else:
        proc = shell.Popen(['sbatch --dependency=afterok:' + dependency + ' ' + jobFile], shell = True, stdout = shell.PIPE)
    result = str(proc.communicate()[0].decode('ascii').strip())
    print(result)
    return(result.split(' ')[-1])

while FlagCluster > 0:
    NumberOfClusters
    CountWhile      = CountWhile + 1

    #store the intial estimate of centriod of the specified number of
    #clusters and bands
    InitialCluster  = np.zeros((NumberOfClusters,NumberOfBands))

    #same resultant binary mask is used for all bands
    #reultant map is generated by using temporal criteria(pixel having
    #temporal uncertainty less than 5# is set to zero) for all bands and
    #using pixel count too (pixel generated by using less than 25
    #images is set to zero)
    #BinaryMask = (ImageIn(:,:,1)>0)

    #Find Random Start location
    ##ImageList       = glob.glob(ImageLocation + '/NorthAfrica/L8*')
    ImageList       = [os.path.basename(x) for x in glob.glob(ImageLocation + '/Other/L8*')]
    print(len(ImageList))
    print(NumberOfClusters)
    #y = np.random.randint(0,len(ImageList),NumberOfClusters)
    y = np.random.randn(len(ImageList), NumberOfClusters)

    if override_init == 0:
        #ImageIn = skimage.io.imread(ImageLocation + '/NorthAfrica/' + ImageList[y[NumberOfClusters - 1]])
        for i in range(0,NumberOfClusters):
            ImageIn = skimage.io.imread(ImageLocation + '/NorthAfrica/' + ImageList[y[NumberOfClusters-1]-1])

            BinaryMask = ~ np.isnan(ImageIn[:,:,0])
            #linear indices of nonzero values
            #IndexNonZero = np.nonzero(BinaryMask)[1]
            IndexNonZero = np.argwhere(BinaryMask)

            #randomly select the speficied number of observations from the list of
            #indices without replacement
            #IndexNonZeroSelect = IndexNonZero[np.random.randint(0,len(IndexNonZero),NumberOfClusters)]
            IndexNonZeroSelect = IndexNonZero[np.random.choice(np.arange(0,len(IndexNonZero)-1), NumberOfClusters, replace='false')]

            #row and column of the random selected values
            RowSelect = IndexNonZeroSelect[:,0]
            ColumnSelect = IndexNonZeroSelect[:,1]
            del BinaryMask
            del IndexNonZero
            del IndexNonZeroSelect
            InitialCluster[i,] = ImageIn[RowSelect[i],ColumnSelect[i],0:NumberOfBands]
        save(SummaryLocation + '/' + override_file,'InitialCluster')
        del RowSelect
        del ColumnSelect
    else:
        load(SummaryLocation + '/' + override_file)
        override_init = 0

    FlagIteration = 1

    while FlagIteration > ThresholdIteration:
        #NumberOfClusters
        #count the number of pixels in each cluster

        ##   subjobs
        #Clean up Cache Folder
        delete(SaveLocation,'I*')
        delete(SaveLocation,'S*')
        delete(LogLocation,'*')

        #sleep(5)
        print('Start the hard work')
        if local_version == 1:
            directories = sorted(glob.glob(ImageLocation + '/*'))
            #for directory in range(0,len(directories)):
            for directory in range(2, directories[0]):
                ImageList = [os.path.basename(x) for x in glob.glob(directory + '/L8*')]
                ImageList = sorted(glob.glob(directory + '/L8*'))
                ImageLocation_Sub = directory
                for chip in ImageList:
                    print(dateNow() + ': '+ str(directories.index(directory) + 1) + ' ' + directory + ': Image ' + str(ImageList.index(chip) + 1) + ' of ' + str(len(ImageList)))
                    chipClassify(ImageLocation_Sub,SaveLocation,chip,NumberOfClusters,InitialCluster)
                #parfor chip=1:size(ImageList,1) #[100,105]
                #    print([dateNow(),': ',str(directory),' ',directories(directory).name,': Image ',str(chip),' of ',str(size(ImageList,1))])
                #    Chip_Classify(ImageLocation_Sub,SaveLocation,fullfile(ImageLocation_Sub,ImageList(chip).name),NumberOfClusters,InitialCluster)
                #end
            #end
        else:
            print(dateNow() + ' creating jobs')
            directories = sorted(glob.glob(ImageLocation + '/*'))

            InSize = 0
            #shell.call('scancel --user=' + username)
            ##unix('scancel --user=larry.leigh','-echo')
            Jobs = list()
            JobID = ''
            for directory in range(2, directories[0]):
                newJob = {
                    'ImageList' : sorted(glob.glob(directory + '/L8*')),
                    'ImageLocation_Sub' : directory
                }
                Jobs.append(newJob)
                In = sorted(glob.glob(directory + '/L*'))
                print('Current size = ' + str(len(In)) + ' added to start size of ' + str(InSize))
                InSize = InSize + len(In)
                jobFile = createJob(newJob.get('ImageLocation_Sub'),SaveLocation,newJob.get('ImageList'),NumberOfClusters,InitialCluster,0)
                print(dateNow() + 'starting queue')

                JobID = sbatch(jobFile, JobID)
                #proc = shell.Popen(['sbatch ' + jobFile], shell = True)
                #shell.call('sbatch ' + jobFile)
                ##[s,w] = unix('sbatch job.slurm','-echo')
                #sleep(30)

            tic = time.time()
            notequal = 1
            print(dateNow() + 'Monitoring')
            repeated = 0
            last = 0
            start = last
            retry = 0
            while notequal == 1:
                ##In = dir(fullfile(ImageLocation,'L*'))
                Out = sorted(glob.glob(SaveLocation + '/I*'))
                if InSize == len(Out):
                    notequal = 0
                else:
                    print(dateNow() + ': not equal pausing for 30 seconds. Total to run: ' + str(InSize) + ', Current Completed: ' + str(len(Out)) + ' Started at: ' + str(start) + ' Total Allowed Run = ' + str((InSize/10)*25*NumberOfBands))
                    #print([dateNow(),':not equal pausing for 30seconds. Total to run: ',str(InSize),', Current Complete: ',str(size(Out,1)),' Started at:',str(start),' Total Allowed Run = ',str((InSize/10)*25*NumberOfBands)])
                    toc = time.time() - tic
                    if last == len(Out):
                        if (repeated > 30 and last != start):  #Stop cause of so many repeats
                            notequal = 0
                            save('notCompleteList.mat',['In','Out'])
                            save('JobsSumbitted_finished.mat','Jobs')
                            save('JobsCompleted_finished.mat','Out')
                            if retry < 3:
                                print('Did not complete all jobs, finding missing jobs and resubmitting')
                                findNonCompletedAndResubmit(Jobs,Out,InSize,SaveLocation,NumberOfClusters,InitialCluster) #Creates softlinks to the missing runs....

                                ImageList = sorted(glob.glob('Reprocess_Temp' + '/L8*'))
                                ImageLocation_Sub = 'Reprocess_Temp'
                                In = sorted(glob.glob(ImageLocation_Sub + '/L*'))
                                print('current size = ' + str(len(In)) + ', added to start size of ' + str(InSize))
                                #InSize = InSize+len(In)

                                jobFile = createJob(ImageLocation_Sub,SaveLocation,ImageList,NumberOfClusters,InitialCluster,0)
                                print(dateNow() + ' starting queue')

                                #shell.call('sbatch job.slurm')
                                ##[s,w] = unix('sbatch job.slurm','-echo')
                                #sleep(30)

                                retry = retry + 1
                                repeated = 0
                                notequal = 1
                                start = last
                        else:
                            repeated = repeated + 1
                            print('Repeated: ' + str(repeated))
                    else:
                        repeated = 0
                        last = len(Out)

                    save('JobsSumbitted.mat','Jobs')
                    save('JobsCompleted.mat','Out')
                    if toc > (InSize/10)*25*NumberOfBands:  #Stop because of time
                        notequal = 0
                        save('notCompleteList.mat','In','Out')
                        save('JobsSumbitted_finished.mat','Jobs')
                        save('JobsCompleted_finished.mat','Out')
                    #sleep(30)
            toc = time.time() - tic
        ##   Collect Results
        print(dateNow() + ': collecting results')
        OutputList = sorted(glob.glob(SaveLocation + '/S*'))
        Collected_MeanCluster        = 0   #not really a mean it's more of a summed moved distance, we divide later by the number to get an actual mean....
        Collected_CountClusterPixels = 0
        Collected_ClusterPixelCount  = 0
        Collected_Totalsse           = 0
        Collected_ClusterSdAllBands  = np.zeros((NumberOfClusters,NumberOfBands))
        Collected_ClusterMeanAllBands= np.zeros((NumberOfClusters,NumberOfBands))
        #len(OutputList)
        for chip in OutputList:
            try:
                load(chip)
                Collected_MeanCluster         = Collected_MeanCluster + MeanCluster
                Collected_CountClusterPixels  = Collected_CountClusterPixels + CountClusterPixels
                Collected_ClusterPixelCount   = Collected_ClusterPixelCount + ClusterPixelCount
                Collected_Totalsse            = Collected_Totalsse + Totalsse
                for numcluster in range(0,NumberOfClusters):
                    Collected_ClusterMeanAllBands[numcluster,] = (Collected_ClusterPixelCount[numcluster]*Collected_ClusterMeanAllBands[numcluster,] + ClusterPixelCount[numcluster]*ClusterMeanAllBands[numcluster,])/(Collected_ClusterPixelCount[numcluster]+ClusterPixelCount[numcluster])
                    Collected_ClusterSdAllBands[numcluster,]   = sqrt(((Collected_ClusterPixelCount[numcluster]-1)*Collected_ClusterSdAllBands[numcluster,] + (ClusterPixelCount[numcluster]-1)*ClusterSdAllBands[numcluster,])/(Collected_ClusterPixelCount[numcluster]+ClusterPixelCount[numcluster]-2))
            except:
                print('file not readable')

        #find the new estimate of the center of cluster
        NewCluster = Collected_MeanCluster/Collected_CountClusterPixels#(bsxfun(@rdivide,Collected_MeanCluster',Collected_CountClusterPixels'))'

        #calculate difference with the previous mean
        DiffMean = abs(InitialCluster - NewCluster)

        #find the maximum value of difference of mean between two clusters
        FlagIteration = max(max(DiffMean))
        print('========================================================================================')
        print(dateNow() + ' What is the maxiumn a cluster center moved from inital = ' + str(FlagIteration) + ', must be less then ' + str(ThresholdIteration))
        print(dateNow() + ' Current Number of Cluster = ' + str(NumberOfClusters))
        print('========================================================================================')
        #store the new cluster value to the initial cluster value
        InitialCluster = NewCluster
        print('========================================================================================')
        del NewCluster
        date = dateNow('%Y.%m.%d_%I.%M')
        save(BaseName + '_' + date + '_InitialCluster_C' + str(NumberOfClusters),'.mat', 'InitialCluster')
        save(BaseName + '_' + date + '_World_CountClusterPixels_C' + str(NumberOfClusters),'.mat', 'Collected_ClusterPixelCount')
        save(BaseName + '_' + date + '_World_ClusterMeanAllBands_C' + str(NumberOfClusters),'.mat','Collected_ClusterMeanAllBands')
        save(BaseName + '_' + date + '_World_ClusterSdAllBands_C' + str(NumberOfClusters),'.mat',  'Collected_ClusterSdAllBands')

    del TsseCluster

    TsseAllCluster[NumberOfClusters] = Collected_Totalsse ##ok<SAGROW>
    ThresholdTsseChange = 20

    #calculate the change in TSSE when number of clusters is increased

    if(NumberOfClusters > 2):
        DiffTssePercentage = ((TsseAllCluster[NumberOfClusters-1] - TsseAllCluster[NumberOfClusters])/(TsseAllCluster[NumberOfClusters-1]))*100

        if(abs(DiffTssePercentage) > ThresholdTsseChange):
            FlagCluster = 1
        else:
            FlagCluster = 0
    NumberOfClusters = NumberOfClusters + 1
date = dateNow('%Y.%m.%d_%I.%M')
save(BaseName + '_' + date + '_World_CountClusterPixels.mat', 'Collected_ClusterPixelCount')
save(BaseName + '_' + date + '_World_ClusterMeanAllBands.mat','Collected_ClusterMeanAllBands')
save(BaseName + '_' + date + '_World_ClusterSdAllBands.mat',  'Collected_ClusterSdAllBands')
