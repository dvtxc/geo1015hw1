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
    max_x = np.max( array_pts_3d[:, 0] )
    max_y = np.max( array_pts_3d[:, 1] )

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
    #Putting x,y cells together to form raster as tuples
    raster = list(zip(xcells,ycells))
    #Putting x,y cells together to form raster as nested list
    #raster = [list(cell) for cell in zip(xcells,ycells)
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

    nodata_value = -9999
    idw_power = -int(j_idw["power"])
    radius = j_idw["radius"]

    # Convert to NumPy array
    arr_pts_3d = np.array(list_pts_3d)

    # Retrieve bounding box for this dataset
    min_x, min_y, max_x, max_y = bbox( arr_pts_3d )

    # Retrieve raster center points
    # TO-DO, use np-array
    list_raster = raster( list_pts_3d, j_idw )
    arr_raster = np.array( list_raster )

    # Subtract the coordinates between the points of arr_pts_3d and arr_raster.
    # Since we want to calculate the distance of all points, perform an np.outer()
    # operation, i.e. between all points of arrays of different sizes.
    dx = np.subtract.outer(arr_raster[:,0], arr_pts_3d[:,0])
    dy = np.subtract.outer(arr_raster[:,1], arr_pts_3d[:,1])

    # Calculate the euclidian distance between all points as 
    # arr_dist = [[dist_rp1_sp1, dist_rp1_sp2, dist_rp1_sp3, ...],
    #             [dist_rp2_sp1, dist_rp2_sp2, dist_rp2_sp3, ...],
    #             ...]
    arr_dist = np.hypot(dx, dy)

    #  Start with IDW interpolation and store output in zi
    num_rp, num_sp = arr_dist.shape
    zi = np.empty(num_rp)
    
    for i in range(num_rp):
        # For every raster point:

        # Calculate distance from this raster point to sample points
        distances = arr_dist[i, :]

        # What sample points are within the circle
        sp_in_circle = np.where(distances < radius)[0]
        #print(arr_raster[i,:])
        #print(sp_in_circle)
        
        if sp_in_circle.size != 0:
            # Get values from sample points within circle
            values = arr_pts_3d[sp_in_circle, 2]
            #print(values)

            weights = distances[sp_in_circle] ** idw_power
            weights /= weights.sum(axis=0)
            
            zi[i] = np.dot(values.T, weights)
            print(zi[i])
            #print('\n')
        else:
            zi[i] = nodata_value


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
