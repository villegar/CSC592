import numpy as np
import os
import skimage.io
import time
# Node jobs
#@ray.remote
def chipClassify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster):
    tic = time.time()
    print(ImageFile)
    sleep(np.random.beta(1,1)*30)
    ImageIn = skimage.io.imread(ImageFile)
    #Info = geotiffinfo(ImageFile)
    toc = time.time() - tic
    ImageRow, ImageColumn, NumberOfBands = ImageIn.shape
    if NumberOfBands > 8:
        NumberOfBands = NumberOfBands - 1

    # prealocate
    Cluster = np.zeros((ImageRow,ImageColumn,NumberOfClusters))
    CountClusterPixels = np.zeros((NumberOfClusters,1))
    MeanCluster = np.zeros((NumberOfClusters,NumberOfBands))
    EuclideanDistanceResultant = np.zeros((ImageRow,ImageColumn,NumberOfClusters))
    directory = '/tmp/ChipS'
    if not os.path.exists(directory):
        os.makedirs(directory)
    #mkdir('/local/larry.leigh.temp/')
    print('starting big loop')
    toc = time.time() - tic
    for j in range(0,ImageRow):
        %print(str(100*j/ImageRow))

        for k in range(0,ImageColumn):
            temp = ImageIn[j,k,0:NumberOfBands]
            #EuclideanDistanceResultant[j,k,:] = 
            #sqrt(sum((repmat(temp,NumberOfClusters,1) - InitialCluster(:,:)).^2,2));
            #		EuclideanDistanceResultant(j,k,:) = sqrt(sum((repmat(squeeze(ImageIn(j,k,:))',NumberOfClusters,1) - InitialCluster(:,:)).^2,2));
            DistanceNearestCluster = min(EuclideanDistanceResultant(j,k,:));

            for l = 1 : NumberOfClusters

                if( DistanceNearestCluster ~=0)
                    if(DistanceNearestCluster == EuclideanDistanceResultant(j,k,l))
                        CountClusterPixels(l) = CountClusterPixels(l)+1;
                        for m = 1 : NumberOfBands
                            MeanCluster(l,m) = MeanCluster(l,m) + ImageIn(j,k,m);
                        end
                        Cluster(j,k,l) = l;
                    end
                end
            end
        end
    end
    print('finished big loop')
    clear EuclideanDistance; clear EuclideanDistanceResultant;
    ImageDisplay = sum(Cluster,3);
    toc = time.time() - tic
    %% count Pixels in each Cluster
    ClusterPixelCount = zeros(1,NumberOfClusters);
    for i = 1 : NumberOfClusters
        %count the number of pixels in each clusters
        ClusterPixelCount(i) = nnz(Cluster(:,:,i));
    end


    %% calculate TSSE within clusters
    TsseCluster = zeros(1,NumberOfClusters);
    CountTemporalUnstablePixel = 0;
    for j = 1 : ImageRow

        for k = 1 : ImageColumn

            %flag to determine the cluster to which pixel belongs to

            FlagSwitch = max(Cluster(j,k,:));

            %store SSE of related to each pixel
            if FlagSwitch ==0
                CountTemporalUnstablePixel = CountTemporalUnstablePixel+1;
            else
                TsseCluster (FlagSwitch) = TsseCluster(FlagSwitch) +  ...
                    sum((squeeze(ImageIn(j,k,1:NumberOfBands)) - InitialCluster(FlagSwitch,:)').^2);
                %count the number of pixels in each cluster
                %Collected_ClusterPixelCount(FlagSwitch) = Collected_ClusterPixelCount(FlagSwitch) + 1;
            end
        end
    end
    Totalsse = sum(TsseCluster); %#ok<NASGU>
    %% get data for final stats....
    %calculate the spatial mean and standard deviation of each cluster

    ClusterMeanAllBands = zeros(NumberOfClusters,NumberOfBands);
    ClusterSdAllBands = zeros(NumberOfClusters,NumberOfBands);
    print('finished small loop')
    toc = time.time() - tic
    for i = 1 : NumberOfClusters

        Temp = Cluster(:,:,i);

        Temp(Temp == i) = 1;

        MaskedClusterAllBands = bsxfun(@times,Temp,ImageIn(:,:,1:NumberOfBands));

        for j = 1 : NumberOfBands
            %Mean = MaskedClusterAllBands(:,:,j
            Temp = MaskedClusterAllBands (:,:,j);
            TempNonZero = Temp(Temp ~=0 );
            TempNonzeronan = TempNonZero(~isnan(TempNonZero));
            %TempNonnan = Temp(~isnan(Temp));
            FinalClusterMean(j) = mean2(TempNonzeronan); %#ok<AGROW>
            FinalClusterSd(j) = std2(TempNonzeronan);%#ok<AGROW>

        end

        ClusterMeanAllBands(i,:) = FinalClusterMean(1,:);
        ClusterSdAllBands(i,:) = FinalClusterSd(1,:);
    end
    %%
    %Output_Catch/ImageDisplay_./FilteredSaharanImagesL8Version3WithoutPixelCountFilter/L8_BA_R300_V3_Lat0016_Lon0035.tif
    filename=fullfile(SaveLocation,['ImageDisplay_',ImageFile(end-32:end-3),'mat']) %#ok<NOPRT>
    save(filename,'ImageDisplay');

    filename=fullfile(SaveLocation,['ClusterCount',str(NumberOfClusters),'_',ImageFile(end-32:end-4),'.tif']) %#ok<NOPRT>
    geotiffwrite(filename,int8(ImageDisplay),Info.RefMatrix);



    filename=fullfile(SaveLocation,['Stats_',ImageFile(end-32:end-3),'mat']) %#ok<NOPRT>
    save(filename,'MeanCluster','CountClusterPixels','ClusterPixelCount','ClusterMeanAllBands','ClusterSdAllBands','Totalsse');
    print('done!')
    toc = time.time() - tic
