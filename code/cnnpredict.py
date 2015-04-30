# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 22:46:32 2015

@author: fujun
"""


import numpy as np
from scipy import signal as sg
#import matplotlib.pyplot as plt

def pad_size(model):
        patchsz = model[0]['mapsize'][0]
        padsz = np.floor(patchsz/2.0)
        return padsz
        
def pad_img(img, padsz):
        # img H x W * channels        
        img = np.pad(img, ((padsz, padsz),(padsz, padsz), (0,0)), 'symmetric')
        return img

def get_sub_img(img, hDivs, wDivs, ih, iw, divsize, overlap):
        # used for parallel processing
        H, W, Channels = img.shape
        start_h = ih*divsize
        if ih == hDivs-1:
            end_h = H
        else:
            end_h = start_h + divsize + 2*overlap            
        
        start_w = iw*divsize
        if iw == wDivs-1:
            end_w = W
        else:
            end_w = start_w + divsize + 2*overlap
            
        sub_img = img[start_h:end_h, start_w:end_w, :]
        return ((ih, iw), sub_img)

def fast_max_pooling(x, pool_scale):
        # non-overlapping max-pooling and zero pading  
        N, H, W = x.shape
        pool_height = pool_width = pool_scale
        assert H % pool_height == 0
        assert W % pool_height == 0
        x_reshaped = x.reshape(N, H / pool_height, pool_height,
                                 W / pool_width, pool_width)
        out = x_reshaped.max(axis=2).max(axis=3)
        return out

def max_pooling_frag(x, offset, pool_scale):
        start_x, start_y = offset[0], offset[1]
        # non-overlapping max-pooling and zero pading  
        N, H, W = x.shape
        H_out = pool_scale*np.floor((H-start_y)/pool_scale)
        W_out = pool_scale*np.floor((W-start_x)/pool_scale)
        x = x[:, start_y:start_y+H_out, start_x:start_x+W_out]
        return fast_max_pooling(x, pool_scale)

def max_pooling_frags(frags_in, in_offset, k):
         k2 = k*k
         x,y = np.meshgrid(range(k),range(k))
         pooling_offset = np.concatenate((y.reshape(1,k2), x.reshape(1,k2))).T
         ks = k*np.ones(k2).reshape(k2,1)
         pooling_offset = np.concatenate((pooling_offset, ks), axis=1)
         
         frags_out = []
         out_offset = []
         for i in range(len(frags_in)):
             frag_in = frags_in[i]
             for j in range(k2):
                 frags_out.append(max_pooling_frag(frag_in, pooling_offset[j,:], k))
                 new_offset = pooling_offset[j,:].reshape(1,3)
                 if not in_offset: # empty
                     out_offset.append(new_offset)
                 else:
                     curr_offset = np.concatenate((in_offset[i], new_offset))
                     out_offset.append(curr_offset)
                 
         return frags_out, out_offset

def gather_frags(frags, offsets, img_size, patch_size, map_size):
        if len(map_size) > 1: # square maps
            map_size = map_size[0]
        patch_num = img_size - patch_size + 1
        N = frags[0].shape[0]
        frags_reunion = np.zeros((N*map_size*map_size, patch_num[0]*patch_num[1]))
        
        for ifrag in range(len(frags)):
            feat_maps = frags[ifrag]
            offset = offsets[ifrag]
            noffset = offset.shape[0]
            nrows, ncols = feat_maps.shape[1:] - map_size + 1
            for irow in range(nrows):
                for jcol in range(ncols):
                    # get this patch data
                    patch_data = feat_maps[:, irow:irow+map_size, jcol:jcol+map_size].flatten()
                    # found the patch id
                    px, py = jcol, irow
                    for i in range(noffset-1, -1, -1):
                        ox, oy, k = offset[i, :]
                        px = ox + px*k
                        py = oy + py*k
                    
                    patch_id = py*patch_num[1] + px
                    frags_reunion[:, patch_id] = patch_data
                    
        return frags_reunion  

def nonlinear_unit(x, non_linear_type):
        if non_linear_type == 3: # rectifier
            x = np.maximum(x, 0)
        elif non_linear_type == 4: # softmax
            xe = np.exp(x)
            x = np.divide(xe, np.sum(xe, axis=0))
        else:
            print "not used now"
        
        return x
         
def predictImage(model, img, padding = False):
        frags, offset = [], []
        for i in xrange(len(model)):
            layer = model[i]
            #print 'processing layer %s ...' %layer['type']
            #print 'layer %s' %layer['type']
            if layer['type'] == 'i':
                if padding:
                    img = pad_img(img, pad_size(model))
                x = np.transpose(img, (2,0,1))
                imgChannels, imgH, imgW = x.shape
                imgsize = np.array([imgH, imgW])
                patchsize = layer['mapsize']
                patchNum = imgsize - patchsize + 1
                frags.append(x)
            elif layer['type'] == 'c': # convolutional layer 
                outputmaps = layer['outputmaps']
                inputmaps  = model[i-1]['outputmaps']
                kernelsize = layer['kernelsize']
            
                k = layer['k']
                b = layer['b']
                
                out_frags = []
                for frag in frags:
                    fraginmaps, fraginH, fraginW = frag.shape
                    assert fraginmaps == inputmaps
                    fragoutH, fragoutW = fraginH - kernelsize + 1, fraginW - kernelsize + 1
                    a = np.zeros((outputmaps, fragoutH, fragoutW))    
                    for kout in xrange(outputmaps):
                        for kin in xrange(inputmaps):
                            a[kout, :, :] += sg.convolve(frag[kin, :, :], k[kout, kin, :, :], 'valid')
                        a[kout, :, :] = nonlinear_unit(a[kout, :, :] + b[kout], layer['nonlinear'])
                    # new frag
                    out_frags.append(a)
                frags = out_frags
                
            elif layer['type'] == 's': # pooling layer
                frags, offset = max_pooling_frags(frags, offset, layer['scale'])
            elif layer['type'] == 'f' or layer['type'] == 'o':
                if model[i-1]['type'] != 'f': # column vector
                   rsp = gather_frags(frags, offset, imgsize, patchsize, model[i-1]['mapsize'])
                
                
                rsp = nonlinear_unit(np.dot(layer['W'], rsp) + layer['b'].reshape(layer['W'].shape[0], 1), layer['nonlinear'])
            else:
                print 'not supported currently'
        # reshape into 2d array        
        rsp = rsp[1, :].reshape(patchNum[0], patchNum[1])
        return rsp        

def collect_subedgemaps(H0, W0, sub_edgemaps, hDivs, wDivs, divsize):
        # pad image first
        edgemap = np.zeros((H0,W0))
        for sub_edgemap_ret in sub_edgemaps:
            (ih, iw) = sub_edgemap_ret[0]
            sub_edgemap = sub_edgemap_ret[1]
            sub_H, sub_W = sub_edgemap.shape
            edgemap[ih*divsize:ih*divsize+sub_H, iw*divsize:iw*divsize+sub_W] = sub_edgemap
        
        return edgemap