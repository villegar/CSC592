import os
from time import sleep
def createJob(ImageLocation,SaveLocation,ImageList,NumberOfClusters,InitialCluster,fileORlist):
    print('Creating job')
    #print(len(ImageList))
    #print(ImageList)
    length = '{}%{:3.0f}'.format(len(ImageList),10+len(ImageList)/32)
    log_name = '%A_%a'

    if(len(ImageList) == 0):
        return

    jobDirectory = 'jobs'
    jobPrefix = os.path.split(ImageLocation)[0].split('/')[-1]
    if not os.path.exists(jobDirectory):
        os.makedirs(jobDirectory)

    #create job
    with open(jobDirectory + '/' + jobPrefix + '.slurm', 'a') as jobFile:
        jobFile.write('#!/bin/bash\n')
        jobFile.write('\n')
        jobFile.write('#SBATCH --job-name=ChipS160      # Job name\n')
        jobFile.write('#SBATCH --nodes=1                # Number of nodes\n')
        jobFile.write('#SBATCH --ntasks-per-node=4      # CPUs per node (MAX=40 for CPU nodes and 80 for GPU)\n')
        jobFile.write('#SBATCH --output=logs/ChipLog_{}.log # Standard output (log file)\n'.format(log_name))
        jobFile.write('#SBATCH --partition=compute      # Partition/Queue\n')
        jobFile.write('#SBATCH --time=0:30:00           # Maximum walltime\n')
        jobFile.write('#SBATCH --array 0-{}             # Throttler\n'.format(length))
        jobFile.write('\n')
        jobFile.write('module purge\n')
        jobFile.write('module use /cm/shared/modulefiles_local\n')
        jobFile.write('module load shared\n')
        jobFile.write('module load slurm\n')
        jobFile.write('module load python\n')
        jobFile.write('source activate CSC592\n')
        jobFile.write('module list\n')
        jobFile.write('\n')
        if fileORlist == 0:
            #jobFile.write('files=$(./{}/*)\n'.format(ImageLocation))
            jobFile.write('files=$(find ./{}/*)\n'.format(ImageLocation))
        else:
            jobFile.write('files=$(sed -n "$SLURM_ARRAY_TASK_ID"p {})'.format(ImageLocation))
        jobFile.write('\n')
        jobFile.write('\n')
        jobFile.write('# Diagnostics\n')
        jobFile.write('echo "The length of the files array is"\n')
        jobFile.write('echo "${#files[@]}"\n')
        jobFile.write('\n')
        jobFile.write('#echo "Listing of all elements of files array"\n')
        jobFile.write('#for file in ${files[@]} do\n')
        jobFile.write('#    echo $file\n')
        jobFile.write('#done\n')
        jobFile.write('\n')
        jobFile.write('echo "SLURM_ARRAY_TASK_ID is"\n')
        jobFile.write('echo $SLURM_ARRAY_TASK_ID\n')
        jobFile.write('\n')
        jobFile.write('echo "Listing just the elements in the array with the SLURM_ARRAY_TASK_ID"\n')
        jobFile.write('echo ${files[$SLURM_ARRAY_TASK_ID]}\n')
        jobFile.write('# End Diagnostics\n')

        jobFile.write('\n')

        jobFile.write('echo matlab -r "Chip_Classify(''{}'', ''{}'',''$files[$SLURM_ARRAY_TASK_ID]'',{},['.format(ImageLocation,SaveLocation,str(NumberOfClusters)))
        for c in range(1,NumberOfClusters):
            if InitialCluster.shape[1] == 7:
                jobFile.write('{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}'.format(*InitialCluster[c,]))
            elif InitialCluster.shape[1] == 16:
                jobFile.write('{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}'.format(*InitialCluster[c,]))
        jobFile.write('])exit"\n')

        jobFile.write('matlab -r "Chip_Classify(''{}'', ''{}'',''$files[$SLURM_ARRAY_TASK_ID]'',{},['.format(ImageLocation,SaveLocation,str(NumberOfClusters)))
        for c in range(1,NumberOfClusters):
            if InitialCluster.shape[1] == 7:
                jobFile.write('{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}'.format(*InitialCluster[c,]))
            elif InitialCluster.shape[1] == 16:
                jobFile.write('{:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}, {:f}'.format(*InitialCluster[c,]))
        jobFile.write('])exit"\n')
    sleep(1)