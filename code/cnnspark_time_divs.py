# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:57:27 2015

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
    divsizes = [600, 500, 400, 300, 200, 100]
    #partitions = [1, 2, 4, 8, 16]
    
    # pad image
    padsz = cnnpredict.pad_size(model)
    img = cnnpredict.pad_img(img0, padsz)
    H, W, Channels = img.shape
    
   
    
    timeused = []
    numDivs = []
    for divsize in divsizes:
        hDivs, wDivs = int(np.floor(H/divsize)), int(np.floor(W/divsize))
        divs = []
        for ih in range(hDivs):
             for iw in range(wDivs):
                 divs.append((ih, iw))
            
        distdivs = sc.parallelize(divs).cache()
        tstart = datetime.now()
        sub_edgemaps = distdivs.map(
            lambda (ih,iw):cnnpredict.get_sub_img(img/255.0, hDivs, wDivs, ih, iw, divsize, padsz)).map(
            lambda ((ih, iw), sub_img):((ih, iw), cnnpredict.predictImage(model, sub_img))).collect()
        
        H0, W0, Channels = img0.shape
        edgemap = cnnpredict.collect_subedgemaps(H0, W0, sub_edgemaps, hDivs, wDivs, divsize)
        tend = datetime.now()
        timeused.append((tend - tstart).seconds)
        numDivs.append(hDivs*wDivs)
    
    f = open('divszs.txt', 'w')
    for i in range(len(divsizes)):
        f.write(' '.join([str(numDivs[i]), str(timeused[i])]))
        f.write('\n')
    f.close()
    #plt.imshow(edgemap, cmap = plt.get_cmap('gray'))
    #plt.show()
    
    sc.stop()
