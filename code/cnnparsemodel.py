# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:41:31 2015

@author: fujun
"""


import scipy.io as sio
import numpy as np
            
def parse_cnn_layer(mlayer):
        dict = {}
        layer_type = mlayer['type'][0,0][0]
        dict['type'] = layer_type
        
        
        if layer_type == 'i':
            # this input layer
            dict['outputmaps'] = mlayer['outputmaps'][0,0][0,0]
            dict['mapsize'] = mlayer['mapsize'][0,0][0]
            #pass
        elif layer_type == 'c':
            # this convolutional layer        
            kernelsize = mlayer['kernelsize'][0,0][0,0]
            dict['kernelsize'] = kernelsize
            dict['nonlinear'] = mlayer['nonlinear'][0,0][0,0]
            dict['neurons'] = mlayer['neurons'][0,0][0,0]
            dict['outputmaps'] = mlayer['outputmaps'][0,0][0,0]
            dict['mapsize'] = mlayer['mapsize'][0,0][0]
            # parse weight k 
            matk = mlayer['k'][0,0]
            n_out, n_in = len(matk[0]), len(matk[0,0][0])
            k = np.zeros((n_out, n_in, kernelsize, kernelsize))
            for i in xrange(n_out):
                ki = matk[0,i]
                for j in xrange(n_in):
                    k[i,j,:,:] = ki[0,j]
            # parse b
            matb = mlayer['b'][0,0]
            b = np.zeros((n_out,1))
            for i in xrange(n_out):
                b[i] = np.squeeze(np.asarray(matb[0,i]))
            
            dict['k'] = k
            dict['b'] = b
        elif layer_type == 's':
            # this is pooling layer
            dict['neurons'] = mlayer['neurons'][0,0][0,0]
            dict['scale'] = mlayer['scale'][0,0][0,0]
            dict['outputmaps'] = mlayer['outputmaps'][0,0][0,0]
            dict['mapsize'] = mlayer['mapsize'][0,0][0]
        elif layer_type == 'f' or layer_type == 'o':
            # this is fully connected layer
            dict['nonlinear'] = mlayer['nonlinear'][0,0][0,0]
            dict['neurons'] = mlayer['neurons'][0,0][0,0]
            # parse W
            dict['W'] = mlayer['W'][0,0]
            # parse b
            dict['b'] = np.squeeze(mlayer['b'][0,0])
        else:
            # layer not supported
            print "layer %s is not supported now." %layer_type
            exit(1)
        return dict
    
def load_matcnn(cnnmat_path):
        
        cnnmat = sio.loadmat(cnnmat_path)
        layers = cnnmat['cnn']['layers'][0,0]
        model = []
        for i in xrange(len(layers[0])):
            #print "parsing layer %d" %i
            model.append(parse_cnn_layer(layers[0,i]))
            
        return model
