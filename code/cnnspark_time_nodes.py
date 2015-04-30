# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:16:08 2015

@author: fujun
"""


import sys
import numpy as np
import cnnpredict
import cnnparsemodel
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import misc

from pyspark import SparkContext



if __name__ == "__main__":
#    if len(sys.argv) != 5:
#        print >> sys.stderr, "Usage: cnnspark <modelpath> <imgpath> <divsize> <partitions>"
#        exit(-1)
    
    sc = SparkContext(appName = "cnnspark", pyFiles=['cnnpredict.py', 'cnnparsemodel.py'])
    
#    model = cnnparsemodel.load_matcnn(sys.argv[1])
#    img0 = misc.imread(sys.argv[2])
#    divsize = int(sys.argv[3])
#    partitions = int(sys.argv[4])
    
    model = cnnparsemodel.load_matcnn('muscle-caffe-20.mat')
    img0 = misc.imread('test.jpg')
    divsize = 200
    partitions = [8, 16]
    
    # pad image
    padsz = cnnpredict.pad_size(model)
    img = cnnpredict.pad_img(img0, padsz)
    H, W, Channels = img.shape
    hDivs, wDivs = int(np.floor(H/divsize)), int(np.floor(W/divsize))
    divs = []
    for ih in range(hDivs):
        for iw in range(wDivs):
            divs.append((ih, iw))
    
    timeused = []
    for partition in partitions:    
        distdivs = sc.parallelize(divs, partition).cache()
        tstart = datetime.now()
        sub_edgemaps = distdivs.map(
            lambda (ih,iw):cnnpredict.get_sub_img(img/255.0, hDivs, wDivs, ih, iw, divsize, padsz)).map(
            lambda ((ih, iw), sub_img):((ih, iw), cnnpredict.predictImage(model, sub_img))).collect()
        
        H0, W0, Channels = img0.shape
        edgemap = cnnpredict.collect_subedgemaps(H0, W0, sub_edgemaps, hDivs, wDivs, divsize)
        tend = datetime.now()
        timeused.append((tend - tstart).seconds)
        print '**** Totally, it took spark %d seconds  for %d partitions ****' %((tend - tstart).seconds, partition)
    
    f = open('nodes.txt', 'w')
    for i in range(len(partitions)):
        f.write(' '.join([str(partitions[i]), str(timeused[i])]))
        f.write('\n')
    f.close()
    #plt.imshow(edgemap, cmap = plt.get_cmap('gray'))
    #plt.show()
    
    sc.stop()
