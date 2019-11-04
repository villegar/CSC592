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
#from numpy import zeros
from datetime import datetime
from PIL import Image
from time import sleep

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


directories         = glob.glob(ImageLocation + '/*')
#
# for directory = 3:size(directories,1)
#     ImageList   = dir(fullfile(ImageLocation,directories(directory).name,'L8*'));
#     ImageLocation_Sub = fullfile(ImageLocation,directories(directory).name) ##ok<NOPTS>
#     In = dir(fullfile(ImageLocation_Sub,'L*'));
#     display(['current size = ',num2str(size(In,1)),', added to start size of ',num2str(InSize)])
#     InSize = InSize+size(In,1);
#
#     create_job(ImageLocation_Sub,SaveLocation,ImageList,NumberOfClusters,InitialCluster)
#     display([datestr(now),'starting queue'])
#     unix('sbatch job.slurm','-echo');
#     pause(15)
# end

#ImageList           = glob.glob(ImageLocation + '/NorthAfrica/L8*')
ImageList           = [os.path.basename(x) for x in glob.glob(ImageLocation + '/NorthAfrica/L8*')]

#start with the number of clusters
NumberOfClusters    = 160;
NumberOfBands       = 16;

#flag to check whether to increase clusters or not
FlagCluster         = 1;
ThresholdIteration  = 0.0001; #0.0001;
CountWhile          = 0;

#local loops in serial, set to 0 to run cluster jobs
local_version       = 0 ##ok<NOPTS>

InitialCluster      = np.empty((0,NumberOfBands), dtype=float32)
username            = 'Roberto.VillegasDiaz'

# L8_BA_R030_V4_Lat0030_Lon0006.tif (270)[793]
@ray.remote
def chipCLassify(ImageLocation_Sub,SaveLocation,ChipPath,NumberOfClusters,InitialCluster):
    print('TEST')

def dateNow():
    return(str(datetime.now()))

def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[0])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)

def loadEnvironment(filename):
    existingShelf = shelve.open(filename)
    for key in existingShelf:
        globals()[key]=existingShelf[key]
    existingShelf.close()
    # Reference: https://stackoverflow.com/questions/2960864/how-can-i-save-all-the-variables-in-the-current-python-session

def removeFiles(directory, pattern):
    # Get a list of all the file paths that math pattern inside directory
    fileList = glob.glob(directory + '/' + pattern)

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print('Error while deleting file : '', filePath)
    # Reference: https://thispointer.com/python-how-to-remove-files-by-matching-pattern-wildcards-certain-extensions-only/

def saveEnvironment(filename, variables):
    newShelf = shelve.open(filename,'n') # 'n' for new
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

while FlagCluster > 0:
    NumberOfClusters
    CountWhile      = CountWhile + 1 ##ok<NOPTS>

    #store the intial estimate of centriod of the specified number of
    #clusters and bands
    InitialCluster  = np.zeros((NumberOfClusters,NumberOfBands))

    #same resultant binary mask is used for all bands
    #reultant map is generated by using temporal criteria(pixel having
    #temporal uncertainty less than 5# is set to zero) for all bands and
    #using pixel count too (pixel generated by using less than 25
    #images is set to zero)
    #BinaryMask = (ImageIn(:,:,1)>0);

    #Find Random Start location
    ##ImageList       = glob.glob(ImageLocation + '/NorthAfrica/L8*')
    ImageList       = [os.path.basename(x) for x in glob.glob(ImageLocation + '/NorthAfrica/L8*')]
    #y = rand.sample(range(1,len(ImageList)+1),NumberOfClusters)
    #y = np.random.choice(len(ImageList),NumberOfClusters)
    y = np.random.randint(0,len(ImageList),NumberOfClusters)

    if override_init == 0:
        for i in range(0,NumberOfClusters):
            ImageIn = skimage.io.imread(ImageLocation + '/NorthAfrica/' + ImageList[y[NumberOfClusters - 1]])

            BinaryMask = ~ np.isnan(ImageIn[:,:,0])
            #linear indices of nonzero values
            #IndexNonZero = np.nonzero(BinaryMask)[1]
            IndexNonZero = np.argwhere(BinaryMask)

            #randomly select the speficied number of observations from the list of
            #indices without replacement
            #IndexNonZeroSelect = IndexNonZero[np.random.choice(len(ImageList),NumberOfClusters-1, replace=False)]
            #IndexNonZeroSelect = IndexNonZero[np.random.choice(NumberOfClusters-1,len(ImageList), replace=False)]
            IndexNonZeroSelect = IndexNonZero[np.random.randint(0,len(IndexNonZero),NumberOfClusters)]
            #rows = np.random.randint(1,len(ImageList),NumberOfClusters)
            #cols = np.random.randint(1,len(ImageList),NumberOfClusters)
            ###IndexNonZeroSelect = IndexNonZero(datasample(1:length(IndexNonZero),NumberOfClusters,...
            ###    'Replace',false));

            #row and column of the random selected values
            #[RowSelect, ColumnSelect] = np.unravel_index(len(BinaryMask),IndexNonZeroSelect,, order='F')
            #[RowSelect, ColumnSelect] = ind2sub(size(BinaryMask),IndexNonZeroSelect);
            RowSelect = IndexNonZeroSelect[:,0]
            ColumnSelect = IndexNonZeroSelect[:,1]
            del BinaryMask
            del IndexNonZero
            del IndexNonZeroSelect
            InitialCluster[i,] = ImageIn[RowSelect[i],ColumnSelect[i],0:NumberOfBands]
            #np.append(InitialCluster, ImageIn[RowSelect[i],ColumnSelect[i],0:NumberOfBands], axis=0)
    else:
        loadEnvironment(SummaryLocation + '/' + override_file)
        override_init = 0
        #InitialCluster

    del RowSelect
    del ColumnSelect

    FlagIteration = 1;

    while FlagIteration > ThresholdIteration:
        #NumberOfClusters;
        #count the number of pixels in each cluster

        ##   subjobs
        #Clean up Cache Folder
        removeFiles(SaveLocation,'I*')
        removeFiles(SaveLocation,'S*')
	    removeFiles(LogLocation,'*')

        sleep(5)
        print('Start the hard work')
        if local_version == 1:
            directories = sorted(glob.glob(ImageLocation + '/*'))
            #for directory in range(0,len(directories)):
            for directory in directories):
                ImageList = [os.path.basename(x) for x in glob.glob(directory + '/L8*')]
                ImageList = sorted(glob.glob(directory + '/L8*'))
                ImageLocation_Sub = directory
                for chip in ImageList:
                    print(dateNow() + ': '+ str(directories.index(directory) + 1) + ' ' + directory + ': Image ' + str(ImageList.index(chip) + 1) + ' of ' + str(len(ImageList)))
                    chipCLassify(ImageLocation_Sub,SaveLocation,chip,NumberOfClusters,InitialCluster)
                #parfor chip=1:size(ImageList,1) #[100,105]
                #    display([datestr(now),': ',num2str(directory),' ',directories(directory).name,': Image ',num2str(chip),' of ',num2str(size(ImageList,1))])
                #    Chip_Classify(ImageLocation_Sub,SaveLocation,fullfile(ImageLocation_Sub,ImageList(chip).name),NumberOfClusters,InitialCluster)
                #end
            #end
        else:
            print(dateNow() + ' creating jobs')
            directories = sorted(glob.glob(ImageLocation + '/*'))

            InSize = 0;
            shell.call('scancel --user=' + username)
            #unix('scancel --user=larry.leigh','-echo');
            for directory in directories):
            #for directory = 3:size(directories,1)
                Jobs(directory).ImageList   = dir(fullfile(ImageLocation,directories(directory).name,'L8*'));
                Jobs(directory).ImageLocation_Sub = fullfile(ImageLocation,directories(directory).name) ##ok<NOPTS>
                In = dir(fullfile(Jobs(directory).ImageLocation_Sub,'L*'));
                display(['current size = ',num2str(size(In,1)),', added to start size of ',num2str(InSize)])
                InSize = InSize+size(In,1);

                create_job(Jobs(directory).ImageLocation_Sub,SaveLocation,Jobs(directory).ImageList,NumberOfClusters,InitialCluster,0)
                display([datestr(now),'starting queue'])

                [s,w] = unix('sbatch job.slurm','-echo');
                pause(30)

            end

            tic
            notequal =1;
            display([datestr(now),'Monitoring'])
            repeated=0;
            last=0;
            start=last;
            retry=0
            while notequal == 1
                #In = dir(fullfile(ImageLocation,'L*'));
                Out = dir(fullfile(SaveLocation,'I*'));
                if InSize==size(Out,1)
                    notequal =0;
                else
                    display([datestr(now),':not equal pausing for 30seconds. Total to run: ',num2str(InSize),', Current Complete: ',num2str(size(Out,1)),' Started at:',num2str(start),' Total Allowed Run = ',num2str((InSize/10)*25*NumberOfBands)])
                    toc
                    if last==size(Out,1)
                        if and(repeated >30,last~=start)  #Stop cause of so many repeats
                            notequal = 0;
                            save('notCompleteList.mat','In','Out')
                            save('JobsSumbitted_finished.mat','Jobs')
                            save('JobsCompleted_finished.mat','Out')
                            if retry<3
                                disp('Did not complete all jobs, finding missing jobs and resubmitting')
                                FindNonCompletesAndResubmit(Jobs,Out,InSize,SaveLocation,NumberOfClusters,InitialCluster);#Creates softlinks to the missing runs....

                                ImageList   = dir(fullfile('Reprocess_Temp','L8*'));
                                ImageLocation_Sub = fullfile('Reprocess_Temp') ##ok<NOPTS>
                                In = dir(fullfile(ImageLocation_Sub,'L*'));
                                display(['current size = ',num2str(size(In,1)),', added to start size of ',num2str(InSize)])
                                #InSize = InSize+size(In,1);

                                create_job(ImageLocation_Sub,SaveLocation,ImageList,NumberOfClusters,InitialCluster,0)
                                disp([datestr(now),'starting queue'])

                                [s,w] = unix('sbatch job.slurm','-echo');
                                pause(30)



                                #create_job(Jobs(directory).ImageLocation_Sub,SaveLocation,Jobs(directory).ImageList,NumberOfClusters,InitialCluster,0)
                                retry=retry+1
                                repeated=0
                                notequal=1;
                                start=last
                            end
                        else
                            repeated=repeated+1;
                            display(['Repeated: ',num2str(repeated)])
                        end
                    else
                        repeated = 0;
                        last=size(Out,1);
                    end


                    save('JobsSumbitted.mat','Jobs')
                    save('JobsCompleted.mat','Out')
                    if toc>(InSize/10)*25*NumberOfBands  #Stop cause of time
                        notequal = 0;
                        save('notCompleteList.mat','In','Out')
                        save('JobsSumbitted_finished.mat','Jobs')
                        save('JobsCompleted_finished.mat','Out')
                    end
                    pause(30)
                end
            end
            toc
        end
##   Collect Results
        display([datestr(now),': collecting results']) ##ok<*DPLAYCHAR>
        OutputList = dir(fullfile(SaveLocation,'S*'));
        Collected_MeanCluster        = 0;   #not really a mean it's more of a summed moved distance, we divid later by the number to get an acutal mean....
        Collected_CountClusterPixels = 0;
        Collected_ClusterPixelCount  = 0;
        Collected_Totalsse           = 0;
        Collected_ClusterSdAllBands  = zeros(NumberOfClusters,NumberOfBands);
        Collected_ClusterMeanAllBands= zeros(NumberOfClusters,NumberOfBands);
        size(OutputList)
        for chip = 1:size(OutputList,1)
            try
                load(fullfile(SaveLocation,OutputList(chip).name))
                Collected_MeanCluster         = Collected_MeanCluster + MeanCluster;
                Collected_CountClusterPixels  = Collected_CountClusterPixels +CountClusterPixels;
                Collected_ClusterPixelCount   = Collected_ClusterPixelCount + ClusterPixelCount;
                Collected_Totalsse            = Collected_Totalsse + Totalsse;
                for numcluster = 1:NumberOfClusters
                    Collected_ClusterMeanAllBands(numcluster,:) = (Collected_ClusterPixelCount(numcluster)*Collected_ClusterMeanAllBands(numcluster,:) + ClusterPixelCount(numcluster)*ClusterMeanAllBands(numcluster,:))/(Collected_ClusterPixelCount(numcluster)+ClusterPixelCount(numcluster));
                    Collected_ClusterSdAllBands(numcluster,:)   = sqrt(((Collected_ClusterPixelCount(numcluster)-1)*Collected_ClusterSdAllBands(numcluster,:) + (ClusterPixelCount(numcluster)-1)*ClusterSdAllBands(numcluster,:))/(Collected_ClusterPixelCount(numcluster)+ClusterPixelCount(numcluster)-2));
                end
            catch
                disp('file not readable')
            end
        end

##
        #find the new estimate of the center of cluster
        NewCluster = (bsxfun(@rdivide,Collected_MeanCluster',Collected_CountClusterPixels'))';

        #calculate difference with the previous mean
        DiffMean = abs(InitialCluster - NewCluster) ##ok<NOPTS>

        #find the maximum value of difference of mean between two clusters
        FlagIteration = max(max(DiffMean));
        display('========================================================================================')
        display([datestr(now),' What is the maxiumn a cluster center moved from inital = ',num2str(FlagIteration),', must be less then ',num2str(ThresholdIteration)])
        display([datestr(now),' Current Number of Cluster = ',num2str(NumberOfClusters)])
        display('========================================================================================')
        #store the new cluster value to the initial cluster value
        InitialCluster = NewCluster
        display('========================================================================================')
        clear NewCluster;
        save(fullfile(SummaryLocation,[BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_InitialCluster_C',num2str(NumberOfClusters),'.mat']), 'InitialCluster');
        save(fullfile(SummaryLocation,[BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_World_CountClusterPixels_C',num2str(NumberOfClusters),'.mat']), 'Collected_ClusterPixelCount');
        save(fullfile(SummaryLocation,[BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_World_ClusterMeanAllBands_C',num2str(NumberOfClusters),'.mat']),'Collected_ClusterMeanAllBands');
        save(fullfile(SummaryLocation,[BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_World_ClusterSdAllBands_C',num2str(NumberOfClusters),'.mat']),  'Collected_ClusterSdAllBands');

    end
    clear TsseCluster

    TsseAllCluster(NumberOfClusters) = Collected_Totalsse; ##ok<SAGROW>
    ThresholdTsseChange = 20;

    #calculate the change in TSSE when number of clusters is increased

    if(NumberOfClusters > 2)

        DiffTssePercentage = ((TsseAllCluster(NumberOfClusters-1) - TsseAllCluster(NumberOfClusters))...
            ./(TsseAllCluster(NumberOfClusters-1)))*100 ##ok<NOPTS>

        if(abs(DiffTssePercentage) > ThresholdTsseChange )

            FlagCluster = 1 ##ok<NOPTS>

        else

            FlagCluster = 0 ##ok<NOPTS>

        end
    end

    NumberOfClusters = NumberOfClusters +1;
end

save(fullfile([BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_World_CountClusterPixels.mat']), 'Collected_ClusterPixelCount');
save(fullfile([BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_World_ClusterMeanAllBands.mat']),'Collected_ClusterMeanAllBands');
save(fullfile([BaseName,'_',datestr(now,'yyyy.mm.dd_HH.MM'),'_World_ClusterSdAllBands.mat']),  'Collected_ClusterSdAllBands');
