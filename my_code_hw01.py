#-- my_code_hw01.py
#-- hw01 GEO1015.2020
#-- DANIEL DOBSON
#-- 5152739
#-- DMITRI VISSER
#-- 4279913


#-- import outside the standard Python library are not allowed, just those:
import math
import numpy as np
import scipy.spatial
import startin 
#-----

#-- tmp import by us, will be removed before handing in the assignment
import time


def bbox(list_pts_3d):
    '''
    BOUNDINGBOX
    This function creates a bounding box. 
    Input: list of x, y, z points.
    Output: bbox (min_x, min_y, max_x, max_y)
    '''
    x_pts = [i[0] for i in list_pts_3d]
    y_pts = [i[1] for i in list_pts_3d]
        
    min_x, min_y, max_x, max_y = min(x_pts), min(y_pts), max(x_pts), max(y_pts)
    bbox = min_x, min_y, max_x, max_y
    return bbox

def bbox_np(array_pts_3d):
    # NumPy implementation of bbox
    min_x = np.min( array_pts_3d[:, 0] )
    min_y = np.min( array_pts_3d[:, 1] )
    max_x = np.min( array_pts_3d[:, 0] )
    max_y = np.min( array_pts_3d[:, 1] )

    bbox = min_x, min_y, max_x, max_y
    return bbox

def raster(list_pts_3d, jparams):
    '''
    RASTER
    Input: list of points x, y, z
    Output: [[x1,y1], [x2,y2], ...., [xn, yn]]
    '''
    #Prepare points for raster
    x_pts = [i[0] for i in list_pts_3d]
    y_pts = [i[1] for i in list_pts_3d]
    z_pts = [i[2] for i in list_pts_3d]
    x_pts.sort()
    y_pts.sort()
    #Bbox for raster input, extracting max_x, max_y
    bbox_raster = bbox(list_pts_3d)
    max_x, max_y = bbox_raster[2], bbox_raster[3]
    ###Make raster
    #Making x, y *center* cells for raster.
    xcells = [(i+jparams["cellsize"]/2) for i in x_pts]
    xcells = [cell for cell in xcells if cell < max_x]
    ycells = [(i+jparams["cellsize"]/2) for i in y_pts]
    ycells = [cell for cell in ycells if cell < max_y]
    #Putting x,y cells together to form raster
    raster = []
    for i in xcells:
        for j in ycells:
            raster.append([i,j])
    return raster 

def nn_interpolation(list_pts_3d, j_nn):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with nearest neighbour interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_nn:        the parameters of the input for "nn"
    Output:
        returns the value of the area
 
    """  
    ### This is what the inputs look like
    #print(list_pts_3d)
    #print(j_nn)
    #print(list_pts_3d[0][0])
    #for i in list_pts_3d:
        #print(i[0])
    
    x_pts = [i[0] for i in list_pts_3d]
    y_pts = [i[1] for i in list_pts_3d]
    z_pts = [i[2] for i in list_pts_3d]
    x_pts.sort()
    y_pts.sort()
    
    bbox_nn = bbox(list_pts_3d)
    print(bbox_nn)
    min_x, min_y, max_x, max_y = bbox_nn[0], bbox_nn[1], bbox_nn[2], bbox_nn[3] 
    
    ### Convert list of points to array for preprocessing the input data
    #list_pts = numpy.array(list_pts_3d)
    
    #Lower left corner
    xll, yll = min_x, min_y
    ll = xll, yll
    print('LOWERLEFTCORNER: {}'.format(ll))

    #Center lower left corner, center upper right corner
    cll_x, cll_y  = (xll + j_nn["cellsize"]/2, yll + j_nn["cellsize"]/2)
    cur_x, cur_y = (max_x - j_nn["cellsize"]/2, max_y - j_nn["cellsize"]/2)
    print('cll_x', cll_x, 'cll_y', cll_y)
    print('cur_x', cur_x, 'cur_y', cur_y)
    
    raster_nn = raster(list_pts_3d, j_nn)
    #print('raster', raster(list_pts_3d,j_nn))

     
    #-- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    #kd = scipy.spatial.KDTree(list_pts)
    #print(kd)
    #d, i = kd.query(kd, k=1)
    #print(d, i)
    
    print("File written to", j_nn['output-file'])


def idw_interpolation(list_pts_3d, j_idw):
    """
    !!! TO BE COMPLETED !!!
    !!! DO NOT CHANGE INPUT ARGUMENTS !!!
     
    Function that writes the output raster with IDW
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_idw:       the parameters of the input for "idw"
    Output:
        returns the value of the area
 
    """  

    # Convert to NumPy array
    arr_pts_3d = np.array(list_pts_3d)

    # Retrieve bounding box for this dataset
    min_x, min_y, max_x, max_y = bbox( arr_pts_3d )


    x_pts = pts_arr[:, 0]
    y_pts = pts_arr[:, 0]
    z_pts = pts_arr[:, 0]
    x_pts = np.sort( x_pts )
    y_pts = np.sort( y_pts )

    # print("cellsize:", j_idw['cellsize'])
    # print("radius:", j_idw['radius'])

    #-- to speed up the nearest neighbour us a kd-tree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.html#scipy.spatial.KDTree
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html#scipy.spatial.KDTree.query
    # kd = scipy.spatial.KDTree(list_pts)
    # i = kd.query_ball_point(p, radius)
    
    print("File written to", j_idw['output-file'])


def tin_interpolation(list_pts_3d, j_tin):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with linear in TIN interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_tin:       the parameters of the input for "tin"
    Output:
        returns the value of the area
 
    """  
    #-- example to construct the DT with scipy
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.Delaunay.html#scipy.spatial.Delaunay
    # dt = scipy.spatial.Delaunay([])

    #-- example to construct the DT with startin
    # minimal docs: https://github.com/hugoledoux/startin_python/blob/master/docs/doc.md
    # how to use it: https://github.com/hugoledoux/startin_python#a-full-simple-example
    # you are *not* allowed to use the function for the tin linear interpolation that I wrote for startin
    # you need to write your own code for this step
    # but you can of course read the code [dt.interpolate_tin_linear(x, y)]
    
    print("File written to", j_tin['output-file'])


def kriging_interpolation(list_pts_3d, j_kriging):
    """
    !!! TO BE COMPLETED !!!
     
    Function that writes the output raster with ordinary kriging interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_kriging:       the parameters of the input for "kriging"
    Output:
        returns the value of the area
 
    """  
    
    
    print("File written to", j_kriging['output-file'])
