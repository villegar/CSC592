import glob
import numpy as np
import os
import rasterio as rio
import earthpy as et
import psutil
import random as rand
import ray
import shelve
import skimage.io
import subprocess as shell
import time
#from numpy import zeros
from datetime import datetime
from math import sqrt
from PIL import Image
from time import sleep
#for image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def Chip_Classify(ImageLocation,SaveLocation,ImageFile,NumberOfClusters,InitialCluster)
	tic = time.time()
	Image.show(title=ImageFile)
	#pause(random('beta',1,1)*30)
	
	ImageIn = mpimg.imread(ImageFile)
	with rio.open(ImageFile) as gtf_img:
		info = gtf_img.meta
	toc = time.time()