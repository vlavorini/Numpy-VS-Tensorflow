def dist_matrix(points):
    u'''
    if expanded, the following is equal to:
    
    expd=np.expand_dims(points,2)
    tiled=np.tile(expd, points.shape[0])
    trans=np.transpose(points)
    num=np.sum(np.square(trans-tiled), axis=1)
    #num
    den1=1-np.sum(np.square(points),1)
    dend=np.expand_dims(den1,1)
    den1M=np.matrix(dend)
    den=den1M * den1M.T
    
    return np.arccosh(1+2*np.divide(num, den))
    '''
    #return np.arccosh(1+2*np.divide(np.sum(np.square(np.transpose(points)-np.tile(np.expand_dims(points,2), points.shape[0])), axis=1), np.matrix(np.expand_dims(1-np.sum(np.square(points),1),1)) * np.matrix(np.expand_dims(1-np.sum(np.square(points),1),1)).T))
    return 1+2*np.divide(np.sum(np.square(np.transpose(points)-np.tile(np.expand_dims(points,2), points.shape[0])), axis=1), np.matrix(np.expand_dims(1-np.sum(np.square(points),1),1)) * np.matrix(np.expand_dims(1-np.sum(np.square(points),1),1)).T)
