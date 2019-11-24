def FindNonCompletesAndResubmit(Jobs,Out,InSize,SaveLocation,NumberOfClusters,InitialCluster)
# SaveLocation = './test'
# NumberOfClusters = 160
# InitialCluster = ones(16,160);

import subprocess as shell
import glob
import numpy as np
import os
import psutil
import random as rand
import ray
import shelve
import skimage.io
import time

#from numpy import zeros

from datetime import datetime
from math import sqrt
from PIL import Image
from time import sleep

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

JobsLeft=Jobs;
found=0;
print "starting: find missing jobs"
for complete in range(0,len(Out[0])-1):
    #disp(complete)
    for Job in range(2,len(JobsLeft[1])-1):
        #display(Job)
#         for complete=1:size(Out,1)
#             complete
            #for subjob in range(1,len(JobsLeft)[Job].ImageList,1)):
            for subjob in range(0, len(JobsLeft[Job].ImageList[0])-1):
                if !len(JobsLeft[Job].ImageList[1])==0:
                if Out[complete].name[end-33:end-5]==JobsLeft[Job].ImageList[subjob].name[0:end-5]):
                    #disp(['found: ',Out(complete).name(end-32:end-4),' ',JobsLeft(Job).ImageList(subjob).name(1:end-4)])
                    JobsLeft[Job].ImageList[subjob]=[];
                    found=found+1;
                    break
                
print "done sorting"
shell.call("scancel --user=larry.leigh","-echo");
save('JobsRESumbitted_finished.mat','JobsLeft')
#delete(fullfile(LogLocation,'*'))
file="rerun_results.txt";
if exist(file,'file'):
    delete(file)

#fid = fopen('rerun_results.txt','w');
count=0;
shell.call("rm Reprocess_Temp/*")
for directory in range(2, len(JobsLeft[1])-1):
    if !len(JobsLeft[directory].ImageList[1])==0:
        #Jobs(directory).ImageList   = dir(fullfile(ImageLocation,directorys(directory).name,'L8*'));
        #Jobs(directory).ImageLocation_Sub = fullfile(ImageLocation,directorys(directory).name) %#ok<NOPTS>
        
        #InSize = InSize+size(In,1);

        
        #In = dir(fullfile(Jobs(directory).ImageLocation_Sub,'L*'));
        print(["current size = ",str(len(JobsLeft[directory].ImageList[0])),', added to start size of ',num2str(InSize)])
        for result in range(0,len(JobsLeft[directory].ImageList[0])-1):
            #create links to all non-processed results
            shell.call(["ln -s ",fullfile(JobsLeft[directory].ImageList[result].folder,JobsLeft[directory].ImageList[result].name)," Reprocess_Temp/",JobsLeft[directory].ImageList[result].name])
            
            count=count+1;
#             ImageListForSubFolder{1,count}=fullfile(JobsLeft(directory).ImageList(result).folder,JobsLeft(directory).ImageList(result).name);
#             fprintf(fid,'%c',fullfile(JobsLeft(directory).ImageList(result).folder,JobsLeft(directory).ImageList(result).name));
#             fprintf(fid, '\r\n');
            #create_job(ImageSubFolder,SaveLocation,ImageListForSubFolder,NumberOfClusters,InitialCluster)
            #create_job(JobsLeft(directory).ImageLocation_Sub,SaveLocation,JobsLeft(directory).ImageList,NumberOfClusters,InitialCluster)

#fclose(fid);
#ImageLocation = '/gpfs/home/larry.leigh/Chip_ClusterClass'
#ImageListForSubFolder   = fullfile(ImageLocation,'Reprocess_Temp')
#ImageList   = dir(fullfile(ImageLocation,'Reprocess_Temp','L8*'));

#disp([datestr(now),'starting queue'])

#create_job(ImageListForSubFolder,            SaveLocation,ImageList   ,NumberOfClusters,InitialCluster,0)
#creacreate_job(ImageListForSubFolder,            SaveLocation,ImageListForSubFolder'   ,NumberOfClusters,InitialCluster,1)

#[s,w] = unix('sbatch job.slurm','-echo');

