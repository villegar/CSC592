function FindNonCompletesAndResubmit(Jobs,Out,InSize,SaveLocation,NumberOfClusters,InitialCluster)
# SaveLocation = './test'
# NumberOfClusters = 160
# InitialCluster = ones(16,160);


JobsLeft=Jobs;
found=0;
disp('starting: find missing jobs')
for complete=1:size(Out,1)
    #disp(complete)
    for Job=3:size(JobsLeft,2)
        #display(Job)
#         for complete=1:size(Out,1)
#             complete
            for subjob=1:size(JobsLeft(Job).ImageList,1)
                if ~size(JobsLeft(Job).ImageList,2)==0
                if strcmp(Out(complete).name(end-32:end-4),JobsLeft(Job).ImageList(subjob).name(1:end-4))
                    #disp(['found: ',Out(complete).name(end-32:end-4),' ',JobsLeft(Job).ImageList(subjob).name(1:end-4)])
                    JobsLeft(Job).ImageList(subjob)=[];
                    found=found+1;
                    break
                end
                end
            end
        #end
    end
end
disp('done sorting')
unix('scancel --user=larry.leigh','-echo');
save('JobsRESumbitted_finished.mat','JobsLeft')
#delete(fullfile(LogLocation,'*'))
file='rerun_results.txt';
if exist(file,'file')
    delete(file)
end
#fid = fopen('rerun_results.txt','w');
count=0;
unix("rm Reprocess_Temp/*")
for directory = 3:size(JobsLeft,2)
    if ~size(JobsLeft(directory).ImageList,2)==0
        #Jobs(directory).ImageList   = dir(fullfile(ImageLocation,directorys(directory).name,'L8*'));
        #Jobs(directory).ImageLocation_Sub = fullfile(ImageLocation,directorys(directory).name) %#ok<NOPTS>
        
        #InSize = InSize+size(In,1);

        
        #In = dir(fullfile(Jobs(directory).ImageLocation_Sub,'L*'));
        display(['current size = ',num2str(size(JobsLeft(directory).ImageList,1)),', added to start size of ',num2str(InSize)])
        for result=1:size(JobsLeft(directory).ImageList,1)
            #create links to all non-processed results
            unix(['ln -s ',fullfile(JobsLeft(directory).ImageList(result).folder,JobsLeft(directory).ImageList(result).name),' Reprocess_Temp/',JobsLeft(directory).ImageList(result).name])
            
            count=count+1;
#             ImageListForSubFolder{1,count}=fullfile(JobsLeft(directory).ImageList(result).folder,JobsLeft(directory).ImageList(result).name);
#             fprintf(fid,'%c',fullfile(JobsLeft(directory).ImageList(result).folder,JobsLeft(directory).ImageList(result).name));
#             fprintf(fid, '\r\n');
            #create_job(ImageSubFolder,SaveLocation,ImageListForSubFolder,NumberOfClusters,InitialCluster)
            #create_job(JobsLeft(directory).ImageLocation_Sub,SaveLocation,JobsLeft(directory).ImageList,NumberOfClusters,InitialCluster)
        end
    end
    
end
#fclose(fid);
#ImageLocation = '/gpfs/home/larry.leigh/Chip_ClusterClass'
#ImageListForSubFolder   = fullfile(ImageLocation,'Reprocess_Temp')
#ImageList   = dir(fullfile(ImageLocation,'Reprocess_Temp','L8*'));

#disp([datestr(now),'starting queue'])

#create_job(ImageListForSubFolder,            SaveLocation,ImageList   ,NumberOfClusters,InitialCluster,0)
#creacreate_job(ImageListForSubFolder,            SaveLocation,ImageListForSubFolder'   ,NumberOfClusters,InitialCluster,1)

#[s,w] = unix('sbatch job.slurm','-echo');
