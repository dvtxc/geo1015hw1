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
    Output: [(x1,y1), (x2,y2), ...., (xn, yn)] or
            [[x1,y1], [x2,y2], ...., [xn, yn]]
    '''
    #Bbox for raster input, extracting max_x, max_y
    bbox_raster = bbox(list_pts_3d)
    min_x, min_y, max_x, max_y = bbox_raster
    #Lower left corner, and center llc
    xll, yll = min_x, min_y
    rows, cols = math.ceil((max_x-min_x)/jparams["cellsize"]),math.ceil((max_y-min_y)//jparams["cellsize"])
    cll_x, cll_y  = (xll + jparams["cellsize"]/2, yll + jparams["cellsize"]/2)
    cur_x, cur_y = (max_x + jparams["cellsize"]/2, max_y + jparams["cellsize"]/2)

    ###Make raster
    xi = np.arange(cll_x,cur_x,jparams["cellsize"])
    yi = np.flip(np.arange(cll_y,cur_y,jparams["cellsize"]))
    
    #Making x, y *center* cells for raster.
    raster_xy = np.array([[i,j] for i in xi for j in yi])
    return raster_xy, rows, cols, xll, yll

def write_asc(list_pts_3d,int_pts,jparams):
    _,rows,cols,xll,yll = raster(list_pts_3d,jparams)
    cellsize = jparams["cellsize"]
    fh = open(jparams['output-file'], "w")
    fh.write(f"NCOLS {cols}\nNROWS {rows}\nXLLCORNER {xll}\nYLLCORNER {yll}\nCELLSIZE {cellsize}\nNODATA_VALUE {-9999}\n") 
    for i in int_pts:
        fh.write(' '.join(map(repr,i)) + '\n')
    fh.close()


def distance_matrix(arr_raster, arr_pts_3d):
    '''
    DISTANCE_MATRIX

    The advantage of this function is, that the distance between all points is only calculated once.

    Input: 
        arr_raster: an m x 2 np.array() containing m xy-coordinates, i.e. of the raster cell centres
        arr_pts_3d: an n x 2 np.array() containing n xy-coordinates, i.e. of the sample points
    Output:
        an m x n np.array() containing euclidian distance between all combinations of raster and sample points
    '''

    # Subtract the coordinates between the points of arr_pts_3d and arr_raster.
    # Since we want to calculate the distance of all points, perform an np.outer()
    # operation, i.e. between all points of arrays of different sizes.
    dx = np.subtract.outer(arr_raster[:,0], arr_pts_3d[:,0])
    dy = np.subtract.outer(arr_raster[:,1], arr_pts_3d[:,1])

    # Calculate the euclidian distance between all points as 
    # arr_dist = [[dist_rp1_sp1, dist_rp1_sp2, dist_rp1_sp3, ...],
    #             [dist_rp2_sp1, dist_rp2_sp2, dist_rp2_sp3, ...],
    #             ...]
    return np.hypot(dx, dy)


def nn_interpolation(list_pts_3d, j_nn):
    """
    Function that writes the output raster with nearest neighbour interpolation
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_nn:        the parameters of the input for "nn"
    Output:
        returns the value of the area
 
    """  
    ## PREPARATION ##
    # Extract z sample points from input
    z_pts = [i[2] for i in list_pts_3d]
    z_pts = np.array(z_pts)
    
    # Extract cellsize and raster (parameters) from input
    cellsize = j_nn["cellsize"]
    raster_nn, rows, cols, xll, yll = raster(list_pts_3d, j_nn)

    # Transform input list_pts_3d into numpy array
    list_pts_3d_arr = np.array(list_pts_3d)
    # x, y extraction
    xy_list_arr = list_pts_3d_arr[:,[0,1]]

    ## START OF INTERPOLATION ##
    # Building KD Tree, provide index to xy_list_arr 
    # for quick lookup of nearest neighbor of any point
    kd = scipy.spatial.KDTree(xy_list_arr)
    # Create empty list to store interpolated values
    nn_pts = []
    for xy in raster_nn:
        # For every cell in raster (x, y) query the KD Tree
        # to get according index from kd to xy from raster_nn
        _, i = kd.query(xy, k=1)
        # Now for this according index, take the z point (value) 
        # from list z_pts, and append to list nn_pts
        nn_pts.append(z_pts[i])

    # Tranform nn_pts list into array, 
    # reshape to raster according to bbox
    # and transpose so orientation of image is correct
    nn_pts=np.array(nn_pts)
    nn_pts=nn_pts.reshape(int(rows), int(cols))
    nn_pts = np.transpose(nn_pts)

    ## FINAL STEP ##
    # Write result to .asc file
    write_asc(list_pts_3d, nn_pts,j_nn)
    print("File written to", j_nn['output-file'])



def idw_interpolation(list_pts_3d, j_idw):
    """
    Function that writes the output raster with IDW
     
    Input:
        list_pts_3d: the list of the input points (in 3D)
        j_idw:       the parameters of the input for "idw"
    Output:
        returns the value of the area
 
    """  

    ## PREPARATION ##
    # Initialise some constants
    nodata_value = -9999
    idw_power = -int(j_idw["power"])
    radius = j_idw["radius"]

    # Convert to NumPy array
    arr_pts_3d = np.array(list_pts_3d)

    # Retrieve bounding box for this dataset
    min_x, min_y, max_x, max_y = bbox_np( arr_pts_3d )

    # Retrieve raster center points
    # TO-DO, use np-array
    list_raster, rows, cols, xll, yll = raster( list_pts_3d, j_idw )
    arr_raster = np.array( list_raster )

    # Determine what raster points are outside the convex hull
    hull = scipy.spatial.Delaunay( arr_pts_3d[:,0:2] )
    inhull = hull.find_simplex( arr_raster ) >= 0

    # Calculate the euclidian distance between all combinations of sample points and raster centres
    arr_dist = distance_matrix(arr_raster, arr_pts_3d)

    ## START OF INTERPOLATION ##
    #  Start with IDW interpolation and store output in zi
    num_rp, num_sp = arr_dist.shape
    zi = np.empty(num_rp)
    
    for i in range(num_rp):
        # For every raster point:

        # Check whether the raster point lies within the convex hull of the sample points
        if inhull[i]:

            # Get distances from this raster point to sample points
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

            else:
                # The current raster point does not have any sample points in sight
                zi[i] = nodata_value
        
        else:
            # The current raster point does NOT lie within the convex hull

            zi[i] = nodata_value


        print('Performing IDW: {0:6.2f}%'.format(i/num_rp*100), end='\r')


    # print("cellsize:", j_idw['cellsize'])
    # print("radius:", j_idw['radius'])

    zi = zi.reshape(int(rows), int(cols))
    zi = np.transpose(zi)

    write_asc(list_pts_3d, zi, j_idw)
    
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
     
    ## PREPARATION ##
    # Extract z sample points from input
    z_pts = [i[2] for i in list_pts_3d]
    # Transform input list_pts_3d into numpy array
    list_pts_3d_arr = np.array(list_pts_3d)
    # x, y extraction
    xy_list_arr = list_pts_3d_arr[:,[0,1]] 
    # Raster
    raster_tin, rows, cols, xll, yll = raster(list_pts_3d, j_tin)
    raster_tin_arr = np.array(raster_tin)
    # No data
    no_data = -9999

    arr_dist = distance_matrix(raster_tin_arr, list_pts_3d_arr)
    
    # Construct DT
    dt = scipy.spatial.Delaunay(xy_list_arr)
    tin_pts= []
    vertices = []
    # For each coordinate, extract simplex (triangle)
    for xy in raster_tin:
        sim = dt.find_simplex(xy)
        # find_simplex returns -1 when points are outside of the triangulation
        if sim == -1:
            tin_pts.append(no_data)
        elif (not sim):
            tin_pts.append(no_data)
        else:
            # Three points that define the triangle, given by their simplices
            simplices = dt.simplices[sim]
            A, B, C = list_pts_3d_arr[simplices[0]], list_pts_3d_arr[simplices[1]], list_pts_3d_arr[simplices[2]]
        
            # Determine coefficients of barycentric coordinates

            w0 = ((B[1] - C[1]) * (xy[0] - C[0]) + (C[0] -B[0]) * (xy[1] - C[1])) / ((B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]))
            w1 = ((C[1] - A[1]) * (xy[0] - C[0]) + (A[0] -C[0]) * (xy[1] - C[1])) / ((B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]))
            w2 = 1 - w0 - w1

            if w0 == 0 and w1 == 0:
                z_pt = A[2]
            elif w0 == 1:
                z_pt = B[2]
            elif w1 == 1:
                z_pt = C[2]
            else:
                # Compute the height value of point coordinate
                z_pt = (w0 * A[2] + w1 * B[2] + w2 * C[2]) / (w0 + w1 + w2)
            tin_pts.append(z_pt)

    tin_pts = np.array(tin_pts).reshape(rows,cols)
    tin_pts = np.transpose(tin_pts)
    write_asc(list_pts_3d,tin_pts,j_tin)  

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

    ## PREPARATION ##
    # Set up some constants
    nodata_value = -9999
    radius = j_kriging["radius"]
    cellsize = j_kriging["cellsize"]
    
    # Variogram constants
    vsill = 1310
    vrange = 280
    vnugget = 1

    # Convert to NumPy array
    arr_pts_3d = np.array(list_pts_3d)

    # Retrieve bounding box for this dataset
    min_x, min_y, max_x, max_y = bbox_np( arr_pts_3d )

    # Retrieve raster center points
    # TO-DO, use np-array
    list_raster, rows, cols, xll, yll = raster( list_pts_3d, j_kriging )
    arr_raster = np.array( list_raster )

    # Determine what raster points are outside the convex hull
    hull = scipy.spatial.Delaunay( arr_pts_3d[:,0:2] )
    inhull = hull.find_simplex( arr_raster ) >= 0

    # Calculate the euclidian distance for all combinations of sample points and raster cell centres.
    arr_dist = distance_matrix(arr_raster, arr_pts_3d)

    #  Start with Kriging interpolation and store output in zi
    num_rp, num_sp = arr_dist.shape
    zi = np.empty(num_rp)

    # To make visualisation a bit easier in QGIS, define (a bit arbitrary ) clipping values for numerical instabilities
    clip_min = np.min( arr_pts_3d[:,2] ) - np.std( arr_pts_3d[:,2] )
    clip_max = np.max( arr_pts_3d[:,2] ) + np.std( arr_pts_3d[:,2] )


    ## START OF INTERPOLATION ##
    for rp in range(num_rp):
        # For every raster point:

        # Check if raster point lies within the convex hull of the sample points
        if inhull[rp]:

            # Get distances from this raster point to other sample points
            distances = arr_dist[rp, :]

            # What sample points are within the circle?
            sp_in_circle = np.where(distances < radius)[0]
            num_in_circle = sp_in_circle.size
            #print(arr_raster[rp,:])
            #print(sp_in_circle)
            
            if num_in_circle > 0:
                # Get values from sample points within circle
                values = arr_pts_3d[sp_in_circle, 2]
                dists = distances[sp_in_circle]
                #print(values)

                # Calculate Lagrange multiplier matrix
                habs2 = distance_matrix(arr_pts_3d[sp_in_circle,:], arr_pts_3d[sp_in_circle,:])
                A = np.ones((num_in_circle + 1, num_in_circle + 1))
                A[-1,-1] = 0
                for i in range(num_in_circle):
                    for j in range(num_in_circle):
                        habs = habs2[i,j]
                        A[i,j] = vsill * (1 - np.exp( - (3 * habs)**2 / vrange**2 )) + vnugget
                        #A[i,j] = 1/2 * ( values[i] - values[j] )**2

                if np.linalg.det(A) == 0:
                    # We will get a singular matrix
                    # In that case, just take the first one around us.
                    zi[rp] = values[0]
                else:
                    # Apply variogram to d
                    d = vsill * (1 - np.exp( - (3 * dists)**2 / vrange**2 )) + vnugget
                    d = np.append(d, 1)

                    weights = np.linalg.solve(A,d)
                    
                    # Calculate interpolated value
                    ijval = np.dot(values.T, weights[0:-1])

                    # Clip if the result is obviously unrealistic because of numeric instability
                    if ijval < clip_min:
                        zi[rp] = clip_min
                    elif ijval > clip_max:
                        zi[rp] = clip_max
                    else:
                        # Store calculated value
                        zi[rp] = ijval

            else:
                # The current raster point does not have any sample points in sight
                zi[rp] = nodata_value

        else:
            # Current raster point does not lie within the convex hull

            zi[rp] = nodata_value
            

        print('Performing Kriging: {0:6.2f}%'.format(rp/num_rp*100), end='\r')

    zi = zi.reshape(int(rows), int(cols))

    write_asc(list_pts_3d, zi, j_kriging)

    
    print("File written to", j_kriging['output-file'])
