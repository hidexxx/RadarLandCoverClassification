# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:55:27 2017
@author: Y.Gou (Y.Gou@ed.ac.uk)

This code is to classify multi-band remote sensing image at a regional scale using Random Forest algorithm, a supervised classification approach. 

Chris Holden (ceholden@gmail.com) - https://github.com/ceholden explained in detail the steps of Random Forest classification with more information can be found in http://ceholden.github.io/open-geo-tutorial/python/chapter_5_classification.html
This is where the first part of our code is based on. 

We adjusted the process chain to over come the memory issue when random forest is applied for large sacle land cover classification
We also   
"""
from osgeo import gdal, gdal_array
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import math
import pandas as pd
# Tell GDAL to throw Python exceptions, and register all drivers
gdal.UseExceptions()
gdal.AllRegister()


def random_forest(raster_tif,training_tif,output_name,mask_value=-9999, predict_other_flag = False, predict_tif='temp.tif'):
    '''
    Args:
    raster_tif: the image the information of which is used to build the random forest model used for classificaiton. Supported data format is geotif. 
    training_tif: a raster containing all the training data for all classes. Data format: geotif. Should be the same extent and pixel size with raster_tif. 
    output_name: The name of the  generated classification image. 
    mask_value: pixels with this value will be masked out from the model building process. Default is -9999
    predict_other_flag:  if set as False, the classification will run on the whole image where the model is trained;
                         if set as True, the classification will run on another image, specified by the next arg 'predict_tif'
                         default is False
    predict_tif: image to be classified. Default is set to equal to raster_tif 
    
    Example:
    random_forest('Sentinel1_Englandsite.tif', 'Englandsite_aoi.tif', 'Englandsite_classificatoinmap.tif')
    random_forest('Sentinel1_Englandsite.tif', 'Englandsite_aoi.tif', 'Englandsite_classificatoinmap.tif', mask_value=0)
    random_forest('Sentinel1_Englandsite.tif', 'Englandsite_aoi.tif', 'Englandsite_classificatoinmap.tif', mask_value=-9999, predict_other_flag = True, predict_tif= 'Sentinel1_Scotlandsite.tif')
    
    Return:    
    A classification map in .tif            
    '''
    ###############################################################
    # Read in the image to be classified and the training ROI image
    ###############################################################
    img_ds = gdal.Open(raster_tif, gdal.GA_ReadOnly)
    roi_ds = gdal.Open(training_tif, gdal.GA_ReadOnly)
    
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray().astype(np.float)
        
    roi = roi_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
        
    ##############################################################
    #contructing the model 
    ##############################################################
    # exploring, masking and  matching the training data with the image   
    n_samples = (roi > 0).sum() #Find the number of non-zero entries (the number of training data samples)
    print('We have {n} samples'.format(n=n_samples))
    
    labels = np.unique(roi[roi > 0])    # the label for each class
    print('The training data include {n} classes: {classes}'.format(n=labels.size, 
                                                                classes=labels)) # the number of classes in the training data
    X = img[roi > 0, :]  
    y = roi[roi > 0]
    
    print('Our X matrix is sized: {sz}'.format(sz=X.shape))
    print('Our y array is sized: {sz}'.format(sz=y.shape))
    # mask out pixels in the image that will either affect the accuracy of the classificaiton model or in the areas we are not interested in.  
    X[np.isnan(X)]= mask_value
    mask =X[:,1] != mask_value
    X = X[mask]
    y = y[mask]
    
    print('After masking, our X matrix is sized: {sz}'.format(sz=X.shape))
    print('After masking, our y array is sized: {sz}'.format(sz=y.shape))
    
    #model construction 
    rf = RandomForestClassifier(n_estimators=500, oob_score=True)    # Initialize our model with 500 trees
    rf = rf.fit(X, y)     # Fit our model to training data
    
    #Random Forest diagnostics
      ## (1) check out the "Out-of-Bag" (OOB) prediction score:
    print('Our OOB prediction of accuracy is: {oob}%'.format(oob=rf.oob_score_ * 100))    
    
      ## (2) ranking the importance of each band
    bands = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16,17,18,19,20]
    for b, imp in zip(bands, rf.feature_importances_):
        print('Band {b} importance: {imp}'.format(b=b, imp=imp))
      ## (3) Cross-tabulate predictions
    df = pd.DataFrame()
    df['truth'] = y
    df['predict'] = rf.predict(X)
    print(pd.crosstab(df['truth'], df['predict'], margins=True))
    
    ################################################################
    ## Predict the whole image (can be the same image or a different one) 
    ################################################################
    # read in the image to be predicted     
    if predict_other_flag == False :
        img_ds = img_ds 
    else:
        img_ds = gdal.Open(predict_tif, gdal.GA_ReadOnly)
    
    img = np.zeros((img_ds.RasterYSize, img_ds.RasterXSize, img_ds.RasterCount),
                   gdal_array.GDALTypeCodeToNumericTypeCode(img_ds.GetRasterBand(1).DataType))
    for b in range(img.shape[2]):
        img[:, :, b] = img_ds.GetRasterBand(b + 1).ReadAsArray()
    
    # setting up the output array 
    out_image = np.zeros((img.shape[0] * img.shape[1]))#Make an output image array
    out_shape = (img.shape[0], img.shape[1])
    
    new_shape = (img.shape[0] * img.shape[1], img.shape[2])     # Take our full image, ignore the Fmask band, and reshape into long array for classification

    # process the image as blocks to overcome the memory issue 
    blockSize = 100000
    n_iterations = int(math.ceil((img.shape[0]*img.shape[1]*1.0)/blockSize))
    
    for i in range(n_iterations):
        print 'Doing block %s of %s...'%(str(i+1), str(n_iterations))
        
        if i == (n_iterations-1):
            img_as_array = img.reshape(new_shape)[(blockSize*i):]
        else:
            img_as_array = img.reshape(new_shape)[(blockSize*i):(blockSize*(i+1))]
       
        img_as_array[np.isnan(img_as_array)]=-9999
        
        # Now predict for each pixel
        class_prediction = rf.set_params(n_jobs = 1).predict(img_as_array)
        
        if i == (n_iterations-1):
            out_image[(blockSize*i):] = class_prediction
        else:
            out_image[(blockSize*i):(blockSize*(i+1))] = class_prediction
    
    out_image = out_image.reshape(out_shape)
    
    # export the classification result as a Geotiff image. 
    driver = gdal.GetDriverByName('GTiff')
    #ds_out = driver.Create('/exports/csce/datastore/geos/groups/SigmaTree/BEIS/Eng_to_Scot_class5_stat6_RFV3.tif',out_shape[1],out_shape[0],1,gdal.GDT_Byte)
    ds_out = driver.Create(output_name,out_shape[1],out_shape[0],1,gdal.GDT_Byte)
    ds_out.SetGeoTransform(img_ds.GetGeoTransform())
    ds_out.SetProjection(img_ds.GetProjection())
    ds_out.GetRasterBand(1).WriteArray(out_image)
    ds_out.FlushCache()
    
