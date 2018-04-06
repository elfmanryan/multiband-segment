from osgeo import gdal
import numpy as np
import cv2
import scipy.stats
from scipy.sparse import csgraph, csr_matrix
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from skimage import segmentation as skseg
from skimage.measure import regionprops
from sklearn import preprocessing
import itertools

BG_VAL = -1

try:
    type(profile)
except NameError:
    def profile(fn): return fn


@profile
def neighbor_matrix(labels,bg=True,connectivity=4,touch=True):
    """
    Generate a connectivity matrix of all labels in the label map.

    Parameters
    ----------
    labels : np.array, shape (M,N)
        The label map.
    bg : bool, optional
        Whether to include the background.
    connectivity : int, optional
        One of [4,8]. If 8, labels also connect via corners.
    touch : bool, optional
        (legacy option) If False, labels are neighbors even if there is a gap of 1 pixel between
        them. (default: True)

    Returns
    -------
    pd.DataFrame, shape (L,L)
        A DataFrame where index and columns are the unique labels and position [i,j] is True iff
        labels i and j are neighbors.
    """
    x = np.unique(labels)
    if not bg: x = x[x != BG_VAL]
    kernels = [
        np.array([[1]]),
        np.array([[1,0,0]]),
        np.array([[0,0,1]]),
        np.array([[1,0,0]]).T,
        np.array([[0,0,1]]).T
    ]
    if connectivity==8:
        kernel.extend([
            np.array([[1,0,0],[0,0,0],[0,0,0]]),
            np.array([[0,0,1],[0,0,0],[0,0,0]]),
            np.array([[0,0,0],[0,0,0],[1,0,0]]),
            np.array([[0,0,0],[0,0,0],[0,0,1]]),
        ])
    shifted = np.stack([cv2.filter2D(labels.astype(np.float32),-1,k) for k in kernels], axis=-1)
    if touch:
        bg_neighbors = shifted[labels!=BG_VAL]
    else:
        bg_neighbors = shifted[labels==BG_VAL]

    bg_neighbors[bg_neighbors==BG_VAL] = np.nan
    bg_neighbors = bg_neighbors[~np.isnan(bg_neighbors).all(axis=1)]
    _mins = np.nanmin(bg_neighbors, axis=1).astype(np.int32)
    _maxs = np.nanmax(bg_neighbors, axis=1).astype(np.int32)
    npairs = np.stack([_mins,_maxs],axis=-1)[_mins != _maxs]
    idx = np.arange(len(x))
    lookup = dict(np.stack([x,idx],axis=-1))
    npairs_idx = np.vectorize(lambda x: lookup[x], otypes=[np.int32])(npairs)
    result = np.zeros((len(x),)*2, dtype=bool)
    result[npairs_idx[:,0],npairs_idx[:,1]] = True
    result[npairs_idx[:,1],npairs_idx[:,0]] = True
    result[x==BG_VAL,:]=False
    result[:,x==BG_VAL]=False
    # DEBUG: Somehow this line is very expensive:
    # result = np.logical_or(result, result.T)
    return pd.DataFrame(result,index=x,columns=x)


@profile
def edge_length(labels,bg=True,connectivity=4):
    """
    Compute the length of an edge between any two labels.

    Parameters
    ----------
    labels : np.array, shape (M,N)
        The label map.
    bg : bool, optional
        Whether to include the background.
    connectivity : int, optional
        One of [4,8]. If 8, labels also connect via corners.

    Returns
    -------
    pd.DataFrame, shape (L,L)
        A DataFrame where index and columns are the unique labels and position [i,j] is the length
        of the edge between labels i and j.
    """
    x = np.unique(labels)
    if not bg: x = x[x != BG_VAL]
    kernels = [
        np.array([[1]]),
        np.array([[1,0,0]]),
        np.array([[0,0,1]]),
        np.array([[1,0,0]]).T,
        np.array([[0,0,1]]).T
    ]
    if connectivity==8:
        kernel.extend([
            np.array([[1,0,0],[0,0,0],[0,0,0]]),
            np.array([[0,0,1],[0,0,0],[0,0,0]]),
            np.array([[0,0,0],[0,0,0],[1,0,0]]),
            np.array([[0,0,0],[0,0,0],[0,0,1]]),
        ])
    shifted = np.stack([cv2.filter2D(labels.astype(np.float32),-1,k) for k in kernels], axis=-1)
    if not bg: shifted[shifted==BG_VAL] = np.nan
    _mins = np.nanmin(shifted, axis=2).astype(np.int32)
    _maxs = np.nanmax(shifted, axis=2).astype(np.int32)
    edge = (_mins != _maxs)
    l1 = _mins[edge]
    l2 = _maxs[edge]
    pairs = np.stack([l1,l2],axis=-1)
    pairs_idx = _replace_labels(pairs, pd.Series(np.arange(len(x)),index=x))
    # fill_index = np.arange(x.min(),x.max() + 1)
    # replace = pd.Series(fill_index,index=fill_index).copy()
    # replace[x] = np.arange(len(x))
    # indexer = np.array([replace[i] for i in range(pairs.min(), pairs.max() + 1)])
    # pairs_idx = indexer[(pairs - pairs.min())]
    result = np.zeros(x.shape*2, dtype=np.int32)
    result[pairs_idx[:,0],pairs_idx[:,1]] += 1
    result[pairs_idx[:,1],pairs_idx[:,0]] += 1
    result /= 2
    return pd.DataFrame(result, index=x, columns=x)


@profile
def _remap_labels(labels, lmap):
    """
    Remaps labels to new values given a mapping l --> l'.

    Parameters
    ----------
    labels : np.array, shape (M,N)
        A map of integer labels.
    lmap : function or pd.Series or dict
        If function, lmap(l) must return the new label.
        If pd.Series or dict, lmap[l] must return the new label.

    Returns
    -------
    np.array, shape (M,N)
        The new label map.
    """
    if callable(lmap):
        indexer = np.array([lmap(i) for i in range(labels.min(), labels.max() + 1)])
    else:
        _lmap = lmap.copy()
        if type(_lmap) is dict:
            _lmap = pd.Series(_lmap)
        # pad lmap:
        fill_index = np.arange(labels.min(),labels.max() + 1)
        replace = pd.Series(fill_index,index=fill_index).copy()
        replace[_lmap.index] = _lmap.values
        indexer = np.array([replace[i] for i in range(labels.min(), labels.max() + 1)])

    return indexer[(labels - labels.min())]


@profile
def merge_connected(labels, connected, touch=False):
    """
    Given a labeled map and a label connectivity matrix, merge connected labels.

    Parameters
    ----------
    labels : np.array, shape (M,N)
        A map of integer labels.
    connected : pd.DataFrame, shape (L,L)
        A DataFrame where index and columns are the unique labels and connected.loc[i,j] == True
        iff labels i and j are connected wrt. any measure.
    touch : bool, optional
        If True, only merge labels that share an edge. (default: False)

    Returns
    -------
    np.array, shape (M,N)
        A new map of labels.
    """
    x = connected.index
    if touch:
        nm = neighbor_matrix(labels,touch=True,bg=False)
        merge = np.logical_and(connected,nm)
    else:
        merge = connected

    csr = csr_matrix(merge)
    cc = csgraph.connected_components(csr, directed=False)
    replace = pd.Series(cc[1], index=x)
    result = _remap_labels(labels,replace)
    return result
    # fill_index = np.arange(labels.min(),labels.max() + 1)
    # replace = pd.Series(fill_index,index=fill_index).copy()
    # replace[x] = cc[1]
    # indexer = np.array([replace[i] for i in range(labels.min(), labels.max() + 1)])
    # return indexer[(labels - labels.min())]


@profile
def bulk_merge(im, labels, mode='distance', threshold=None, touch=True):
    """
    Compute segment distance matrix and merge similar segments.
    """
    if touch:
        nm = neighbor_matrix(labels,touch=True,bg=False)
    else:
        ll = np.unique(labels)
        ll = ll[ll != BG_VAL]
        nm = pd.DataFrame(np.ones((len(ll),len(ll)),dtype=bool), index=ll, columns=ll)

    # ------------------------------------------------------------------------
    if mode == 'distance':
        # distance-based
        stats = patch_stats(im, labels, what=['mean'])
        mean = stats['mean']
        x = mean.index
        if threshold is None: threshold = 50
        # dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(stats['mean'],'euclidean'))
        # similar = (dist < threshold)
        i1,i2 = np.meshgrid(x, x, copy=False)
        p1 = mean.loc[i1[nm]]
        p2 = mean.loc[i2[nm]]
        d = np.linalg.norm(p1.values - p2.values, axis=1)
        similar = np.zeros(nm.shape, dtype=bool)
        similar[nm] = (d < threshold)

    elif mode == 'ttest':
        # t-test based
        stats = patch_stats(im, labels, what=['mean','var','stats'])
        mean = stats['mean']
        var = stats['var']
        area = stats['stats']['area']
        x = mean.index
        if threshold is None: threshold = 0.0001

        eps_var = 0 # Avoid zero variance.
        x1,x2 = np.meshgrid(x, x, copy=False)
        i1 = x1[nm]
        i2 = x2[nm]
        x1_mean = mean.loc[i1].values
        x2_mean = mean.loc[i2].values
        x1_var = var.loc[i1].values + eps_var
        x2_var = var.loc[i2].values + eps_var
        x1_size = np.expand_dims(area.loc[i1].values,axis=-1)
        x2_size = np.expand_dims(area.loc[i2].values,axis=-1)
        t = np.abs(x1_mean - x2_mean) / np.sqrt(x1_var/x1_size + x2_var/x2_size)
        df = (x1_var/x1_size + x2_var/x2_size)**2 / (x1_var**2/(x1_size**2 * (x1_size-1)) + x2_var**2/(x2_size**2 * (x2_size-1)))
        p = 1 - (scipy.stats.t.cdf(t,df) - scipy.stats.t.cdf(-t,df))
        p_ = np.nan_to_num(p).min(axis=1)
        similar_flat = (p_ > threshold)
        similar = np.zeros(nm.shape,dtype=bool)
        similar[nm] = similar_flat

    elif mode == 'shape':
        if threshold is None: threshold = 100
        stats = patch_stats(im, labels, what=['stats','coords'])
        area = stats['stats']['area']
        coords = stats['coords']
        x = area.index
        i1,i2 = np.meshgrid(x, x, copy=False)
        coords1 = coords.loc[i1[nm]]
        coords2 = coords.loc[i2[nm]]
        area1 = area.loc[i1[nm]]
        area2 = area.loc[i2[nm]]
        # Higher merge score if centroids are close, and if area is large
        d = np.linalg.norm(coords1.values - coords2.values, axis=1)
        a = area1.values + area2.values
        score = a / d
        similar = np.zeros(nm.shape, dtype=bool)
        similar[nm] = (score > threshold)

    elif mode == 'edge':
        if threshold is None: threshold = 0.01
        e = edge_length(labels,bg=False)
        stats = patch_stats(im, labels, what=['stats'])
        # area = stats['stats']['area']
        peri = stats['stats']['perimeter']
        x = peri.index
        i1,i2 = np.meshgrid(x, x, copy=False)
        # area1 = area.loc[i1[nm]]
        # area2 = area.loc[i2[nm]]
        peri1 = peri.loc[i1[nm]]
        peri2 = peri.loc[i2[nm]]
        edges = e.values[nm]
        score = edges / (peri1.values + peri2.values)
        similar = np.zeros(nm.shape, dtype=bool)
        similar[nm] = (score > threshold)

    elif mode == 'kmeans':
        stats = patch_stats(im, labels, what=['mean','coords'])
        x = stats['mean'].index
        mean = stats['mean']
        # var = stats['var']
        # yxgrid = np.stack(np.mgrid[0:im.shape[1],0:im.shape[0]], axis=2)
        # coords = patch_stats(yxgrid,labels)['mean']
        # coords.columns = ['y','x']
        # X = pd.concat([mean,coords],axis=1)
        X = mean
        # X = pd.concat([mean,var],axis=1)
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clust = KMeans(n_clusters=threshold, random_state=0).fit(X_scaled)
        cl = clust.labels_
        l1,l2 = np.meshgrid(cl,cl,copy=False)
        similar = (l1 == l2)

    np.fill_diagonal(similar, False)

    # ------------------------------------------------------------------------
    if touch: merge = np.logical_and(similar,nm)
    merged_labels = merge_connected(labels, merge)
    # ------------------------------------------------------------------------
    # merged_labels = relabel(merged_labels)
    return merged_labels

    # ------------------------------------------------------------------------
    cv2.imwrite('sample_segments.jpg',colorize(merged_labels))
    # ------------------------------------------------------------------------


    # multivariate t-test?
    # cov = stats['cov']
    x1_cov,x2_cov = np.meshgrid(stats['cov'],stats['cov'],copy=False)
    cov = (x1_size-1) * x1_cov + (x2_size-1) * x2_cov
    d_mu = x1_mean - x2_mean
    Tsq = x1_size * x2_size / (x1_size + x2_size) * d_mu * cov * d_mu
    # ...
    from spm1d.stats import hotellings2
    _data2 = pd.Series(index=ll,dtype=np.object)
    for l,r in _data.iterrows(): _data2.loc[l] = np.stack(r,axis=-1)
    for l,r in _data2.iteritems(): _data2.loc[l] = r.T
    T2 = hotellings2(_data2.loc[2],_data2.loc[3])
    _data3 = np.expand_dims(_data2,axis=-1)
    def hdist(u,v):
        try:
            return hotellings2(u[0],v[0]).inference(0.01).h0reject
        except:
            return np.nan

    T2_grid = scipy.spatial.distance.pdist(_data3, hdist)



# @profile
def _fill_gaps(labels):
    """
    Bridge patches with identical label.
    """
    new_labels = labels.copy()
    kernels = [
        np.array([[1,0,0],[0,0,0],[0,0,0]]),
        np.array([[0,0,1],[0,0,0],[0,0,0]]),
        np.array([[0,0,0],[0,0,0],[1,0,0]]),
        np.array([[0,0,0],[0,0,0],[0,0,1]]),
        np.array([[1,0,0]]),
        np.array([[0,0,1]]),
        np.array([[1,0,0]]).T,
        np.array([[0,0,1]]).T
    ]

    shifted = np.stack([ cv2.filter2D(new_labels.astype(np.float32),-1,k) for k in kernels ], axis=-1)
    shifted[shifted == BG_VAL] = np.nan
    _var = np.nanvar(shifted,axis=2)
    _var[np.isnan(_var)] = 0
    bg = (new_labels == BG_VAL)
    bad = (_var > 0)
    bordering_two = (~np.isnan(shifted[:,:,4:])).sum(axis=2) >= 2
    edge = np.logical_and(bg, np.logical_and(~bad, bordering_two))
    new_labels[edge] = np.nanmedian(shifted[edge,:],axis=1).astype(int)

    return new_labels.astype(np.int32)


def layerwise_segments(im, threshold, bandwidth):
    """
    Assigns segments based on iterative thresholding.

    Parameters
    ----------
    im : np.array, shape (M,N)
        The original image.
    threshold : int
        Values below threshold will be regarded as background.
    bandwidth : int
        The image will be thresholded in steps of 'bandwidth'.

    Returns
    -------
    np.array, shape (M,N)
        A label map
    """
    EXPANSION_LOCK = False
    MIN_AREA = 50
    CONNECTIVITY_KERNEL = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    RECT_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    first = True
    segment_labels = np.ones(im.shape) * BG_VAL
    background = np.ones(im.shape).astype(bool)

    ##
    ## Iterate over a series of thresholds, from the highest (strictest) to the lowest (`threshold`)
    ## in steps of `bandwidth`.
    ##
    limit = 256
    while (limit >= threshold):
        ##
        ## Threshold the image.
        ##
        limit -= bandwidth
        active_threshold = max(limit,threshold)
        print(active_threshold)
        mask = (im > active_threshold)

        ##
        ## Do a prefilter on the thresholded image:
        ## Remove some foreground and background noise through
        ## closing and opening, respectively.
        ##
        _thresh = mask.astype(np.uint8)*255
        _thresh = cv2.morphologyEx(_thresh,cv2.MORPH_OPEN,RECT_KERNEL,iterations=2)
        _thresh = cv2.morphologyEx(_thresh,cv2.MORPH_CLOSE,RECT_KERNEL,iterations=2)
        mask = (_thresh > 0)

        ##
        ## Label the resulting patches of contiguous pixels.
        ##
        _, labels = cv2.connectedComponents(mask.astype(np.uint8), 4, cv2.CV_32S)
        labels[~mask] = BG_VAL

        if first:
            ##
            ## In the first iteration, just keep the labels assigned from the connectedComponents analysis.
            ##
            background = ~mask
            segment_labels = labels
            segment_labels[background] = BG_VAL
            first = False
        else:
            ##
            ## In each following iteration, we want to determine which patches to expand and where
            ## to form new patches.
            ## -------------------------------------------------------------------------------------
            ##

            ##
            ## Some patches are "good enough" and don't need to expand further. These will be marked
            ## with an "expansion lock".
            ## --> Areas whose variance would be increased by a large amount.
            ## --> Don't lock very small areas.
            ##
            ## DEBUG: Turned off
            if EXPANSION_LOCK:
                _,patch_mean = patch_stats(im, segment_labels, lambda x: np.percentile(x,5), labels=True)
                patch_mean_diff = patch_mean - (active_threshold + 2*bandwidth)
                expansion_lock = (patch_mean_diff > 0)
                expansion_lock[area_threshold(expansion_lock, 200, labels=False)] = False
                expansion_lock_padded = cv2.dilate(expansion_lock.astype(np.uint8), CONNECTIVITY_KERNEL, iterations=1).astype(bool)

            ##
            ## Distinguish between two cases:
            ## 1) A new seed has been found. (no previous segment present)
            ## 2) A previous segment has been expanded.
            ##
            # Before: background, now: foreground
            mask_new = np.logical_and(background, mask)
            # Before: foreground, now: foreground
            mask_updated = np.logical_and(~background, mask)
            # All labels that cover new pixels:
            labels_new = np.unique(labels[mask_new])
            # All labels that cover existing segments:
            labels_updated = np.unique(labels[mask_updated])
            # All labels that cover new pixels, but also existing segments:
            labels_expanded = np.intersect1d(labels_new, labels_updated)
            # All labels that exclusively cover new pixels:
            labels_created = labels_new[np.logical_not(np.isin(labels_new,labels_updated))]

            mask_expanded = np.isin(labels,labels_expanded)
            mask_created = np.isin(labels,labels_created)

            ##
            ## Very small patches are not allowed to form seeds
            ##
            # Approach (a):
            _thresh = mask_created.astype(np.uint8)*255
            _thresh = cv2.morphologyEx(_thresh,cv2.MORPH_OPEN,RECT_KERNEL,iterations=2)
            mask_created = np.logical_and(mask_created, _thresh > 0)
            # Approach (b):
            # mask_created[area_threshold(mask_created, MIN_AREA, labels=False)] = False

            ##
            ## Update the overall mask to reflect the area threshold.
            ##
            mask = np.logical_or(mask_created, mask_expanded)

            ##
            ## Generate the seeds and the "unknown" mask, i.e. the area into which to expand during
            ## the watershed step.
            ##
            # seeds = np.logical_or(segment_labels != BG_VAL, mask_created).astype(np.uint8)*255
            seeds = (segment_labels != BG_VAL).astype(np.uint8)*255
            unknown = np.logical_and(mask_expanded, seeds == 0)

            ##
            ## Apply expansion lock.
            ## The locked patches do not participate in the watershed round.
            ##
            # seeds[expansion_lock_padded] = 0
            # unknown[expansion_lock_padded] = False
            # mask_expanded[expansion_lock_padded] = False

            ##
            ## From the seeds and the unknown area, generate the markers for the watershed.
            ## Note: Values of 0 mean "unknown" for the watershed algorithm, so we need to shift
            ## the markers.
            ##
            _, markers = cv2.connectedComponents(seeds, 4, cv2.CV_32S)
            markers = markers + 1
            markers[unknown] = 0

            ##
            ## Execute the watershed algorithm.
            ##
            # watershed_map = cv2.cvtColor(cv2.GaussianBlur(im,(101,101),30), cv2.COLOR_GRAY2RGB)
            watershed_map = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            # watershed_map = cv2.cvtColor(mask_expanded.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
            seg = cv2.watershed(watershed_map, markers.copy())

            ## DEBUG: Fix this
            if active_threshold > threshold:    # Skip this during the last iteration
                seg[np.logical_and(mask_expanded, seg==1)] = BG_VAL
            seg[~mask] = BG_VAL


            ##
            ## DEBUG:
            ## Erode AFTER watershed (create an edge between labels).
            ##
            # eroded_labels = erode_labels(seg)
            # erosion_edge = (eroded_labels != seg)

            ##
            ## Identify the area where expansion lock segments have been expanded,
            ## and reverse the expansion.
            ## NOTE: This needs to be done better. Before the actual watershed.
            ##
            ## DEBUG: Turned off
            if EXPANSION_LOCK:
                expansion_locked_labels = np.unique(eroded_labels[expansion_lock])
                illegal_expansion = np.logical_and(np.isin(eroded_labels, expansion_locked_labels), ~expansion_lock)
                eroded_labels[illegal_expansion] = BG_VAL


            ## For newly created segments, decide if the segment is good enough to be accepted.
            # accept_created = np.logical_and(mask_created, stat >= segments[mask_expanded].min())
            # segments[accept_created] = stat[accept_created]

            ##
            ## Update background information
            ##
            background = ~mask # DEBUG
            segment_labels = eroded_labels
            # segment_labels[~mask] = BG_VAL


            ## DEBUG:
            # write = np.stack([mask_expanded, mask_created, seeds > 0],axis=-1).astype(np.uint8)*255
            # cv2.imwrite('seg/sample_banded_{:03d}.jpg'.format(active_threshold),write)

    ##
    ## DEBUG: FINAL STEP
    ##
    # watershed_map = im
    # watershed_map[im < threshold] = 0
    # watershed_map = cv2.cvtColor(watershed_map, cv2.COLOR_GRAY2RGB)
    # markers = segment_labels + 1
    # markers[im < threshold] = markers.max()+1
    # final = cv2.watershed(watershed_map, markers)
    # final[im < threshold] = -1


    ##
    ## Merge all very small patches (from the last step) with their most similar neighbor.
    ##
    segment_labels = relabel(segment_labels)
    segment_labels = merge_small_areas(im, segment_labels, MIN_AREA)

    return segment_labels


@profile
def merge_small(im, labels, threshold):
    """
    Merge small segments with their most similar neighbor.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The multivariate image.
    labels : np.array, shape (M,N)
        The corresponding label map.
    threshold : int
        Segments with an area less than `threshold` will be merged (if they have a neighbor).

    Returns
    -------
    np.array, shape (M,N)
        The new label map.
    """
    new_labels = labels.copy()
    stats = patch_stats(im,new_labels)
    mean = stats['mean']
    area = stats['stats']['area']
    small = (area < threshold)
    nm = neighbor_matrix(new_labels,touch=True,bg=False)

    # find small segments that have at least one neighbor (other than background)
    # n_neighbors = nm.sum(axis=1)

    # Merge the ones that have one neighbor:
    # has_one_neighbor = (n_neighbors == 1)
    # merge_mask = np.logical_and(has_one_neighbor,small)
    # merge_idx = small.index[merge_mask]
    # merge_with = small.index[nm.values[merge_mask,:].argmax(axis=1)]
    # for i,m in zip(merge_idx,merge_with): new_labels[new_labels == i] = m

    # For the others, find closest neighbor
    # has_neighbors = (n_neighbors > 1)
    # has_neighbors = nm.any(axis=1)
    # select = np.logical_and(has_neighbors,small)
    # select = small
    _nm = nm.loc[small,:]
    _i1,_i2 = np.meshgrid(mean.loc[:].index, mean.loc[small].index, copy=False)
    # Compute distances
    p1 = mean.loc[_i1[_nm]]
    p2 = mean.loc[_i2[_nm]] # These are the small ones with neighbors
    d = np.linalg.norm(p1.values - p2.values, axis=1)

    # This line can probably be made faster:
    # For every small patch p: the index of p1 where p2 is p and d is minimum
    df = pd.DataFrame(index=p1.index)
    df['p2'] = p2.index
    df['d'] = d
    # df = df.sort_values(by='d')
    merge = df.groupby('p2')['d'].idxmin()
    new_labels = _remap_labels(new_labels, merge)

    # fill_index = np.arange(new_labels.min(),new_labels.max() + 1)
    # replace = pd.Series(fill_index,index=fill_index).copy()
    # replace[merge.index] = merge
    # # ----
    # indexer = np.array([replace[i] for i in range(new_labels.min(), new_labels.max() + 1)])
    # new_labels = indexer[(new_labels - new_labels.min())]

    # ----
    # for i,m in nn.iteritems(): new_labels[new_labels == i] = m
    # new_labels,_,_ = skseg.relabel_sequential(new_labels,1)
    return new_labels
    # len(np.unique(new_labels))
    cv2.imwrite('sample_segments.jpg',colorize(new_labels))



def plot_categorical_im(im):
    colored = np.zeros(im.shape[:2] + (3,), dtype=np.uint8)
    colors = [ [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48] ]
    for i in np.arange(im.shape[2]): colored[im[:,:,i] > 0] = colors[i]
    cv2.imwrite( 'sample_categorical.jpg', cv2.cvtColor(colored,cv2.COLOR_BGR2RGB) )


def single_band_watershed(band):
    RECT_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    blur = cv2.GaussianBlur(band, ksize=(25,25), sigmaX=1.5)
    # return assign_segments(blur, 5, 30)

    ## Foreground
    thresh_fg = (blur > 30).astype(np.uint8)*255
    thresh_fg = cv2.morphologyEx(thresh_fg,cv2.MORPH_OPEN,RECT_KERNEL,iterations=2)
    thresh_fg = cv2.erode(thresh_fg,RECT_KERNEL,iterations=3)
    ## Background
    thresh_bg = (blur > 0).astype(np.uint8)*255
    # thresh_bg = cv2.morphologyEx(thresh_bg,cv2.MORPH_CLOSE,RECT_KERNEL,iterations=2)
    # thresh_bg = cv2.dilate(thresh_bg,RECT_KERNEL,iterations=5)
    ## Watershed
    unknown = (thresh_bg - thresh_fg) > 0
    _, markers = cv2.connectedComponents(thresh_fg, 4, cv2.CV_32S)
    markers = markers + 1
    markers[unknown] = 0
    seg = cv2.watershed(cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB), markers)
    seg[seg==1] = BG_VAL
    return seg
    # return erode_labels(seg)


def layered_segments(im):
    ndim = im.shape[2]
    thresh = np.arange(10,255,20)
    blurred = cv2.GaussianBlur(im, ksize=(25,25), sigmaX=3)
    RECT_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    CIRC_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    CONNECTIVITY_KERNEL = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    # masks = np.zeros(im.shape[:2] + thresh.shape)
    masks = np.zeros(im.shape,dtype=np.uint8)
    for i in range(ndim):
        contours = (np.atleast_3d(blurred[:,:,i]) > thresh).astype(np.uint8)
        contours = cv2.morphologyEx(contours,cv2.MORPH_OPEN,CIRC_KERNEL,iterations=5)
        contours = cv2.morphologyEx(contours,cv2.MORPH_CLOSE,CIRC_KERNEL,iterations=1)
        # cv2.imwrite('sample_segments.jpg',scale_255(contours.sum(axis=2)))
        masks[:,:,i] = contours.sum(axis=2)
    # masks = masks * (255/masks.max())
    # eroded = masks.copy()
    # eroded[eroded==0] = BG_VAL
    # for i in range(ndim): eroded[:,:,i] = erode_labels(eroded[:,:,i])

    # return masks

    ##
    ## Find local maxima.
    ##
    from skimage.feature import peak_local_max
    maxima = np.zeros(im.shape)
    for i in range(ndim):
        peaks = peak_local_max(masks[:,:,i], min_distance=20, threshold_abs=2, indices=False)
        maxima[peaks,i] = 1

    # c = cv2.cvtColor(scale_255(masks[:,:,2]).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # c[maxima[:,:,2] > 0] = [0,0,255]
    # cv2.imwrite('sample_segments.jpg',c)

    ##
    ## Use the local maxima as seeds for the watershed algorithm.
    ##
    unknown = np.logical_and(masks > 0, maxima == 0)
    segments = np.zeros(im.shape)

    for i in range(ndim):
        seeds = maxima[:,:,i].astype(np.uint8)*255
        _, markers = cv2.connectedComponents(seeds, 4, cv2.CV_32S)
        markers = markers + 1
        markers[unknown[:,:,i]] = 0
        # watershed_map = cv2.cvtColor(masks[:,:,i].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        watershed_map = cv2.cvtColor(blurred[:,:,i].astype(np.uint8), cv2.COLOR_GRAY2RGB)
        segments[:,:,i] = cv2.watershed(watershed_map, markers)

    sneaky_background = np.logical_and(unknown,segments==1)
    true_background = np.logical_and(~unknown,segments==1)
    segments[true_background] = BG_VAL
    ##
    ## What to do with the sneaky background? Leave as is?
    ##
    # segments[sneaky_background] = BG_VAL

    return segments


@profile
def patch_stats(im, labels, what=['mean','stats']):
    """
    Compute statistics for each labeled segment in the image.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The multivariate image.
    labels : np.array, shape (M,N)
        The corresponding label map.
    what : list, optional
        The properties of each segment to compute. Must be a sublist of
        ['mean', 'var', 'cov', 'coords', 'stats']. (default: ['mean','coords'])

    Returns
    -------
    dict of pd.DataFrame
        A dictionary of segment properties as DataFrames. Each attribute passed in `what` will be
        a key in the dictionary.
    """
    _im = np.atleast_3d(im)
    ndim = _im.shape[2]
    ll = np.unique(labels)
    ll = ll[ll != BG_VAL]
    offset = labels[labels != BG_VAL].min() - 1
    _labels = labels - offset
    result = {}
    if 'mean' in what:
        result['mean'] = pd.DataFrame(index=ll, columns=np.arange(ndim))
    if 'var' in what:
        result['var'] = pd.DataFrame(index=ll, columns=np.arange(ndim))
    if 'cov' in what:
        result['cov'] = pd.Series(index=ll,dtype=np.object)
        _data = pd.DataFrame(index=ll, columns=np.arange(ndim))

    for i in range(ndim):
        props = regionprops(_labels,_im[:,:,i])
        if 'mean' in what:
            result['mean'][i] = np.array([_.mean_intensity for _ in props])
        if 'var' in what:
            result['var'][i] = np.array([np.var(_.intensity_image[_.image]) for _ in props])
        if 'cov' in what:
            _data[i] = np.array([_.intensity_image[_.image] for _ in props])

    if 'cov' in what:
        for l,r in _data.iterrows(): result['cov'].loc[l] = np.cov(np.stack(r))

    if 'stats' in what:
        result['stats'] = pd.DataFrame(index=ll)
        result['stats']['area'] = np.array([_.area for _ in props])
        result['stats']['perimeter'] = np.array([_.perimeter for _ in props])

    if 'coords' in what:
        result['coords'] = pd.DataFrame(np.array([_.centroid for _ in props]), index=ll, columns=['x','y'])

    return result


@profile
def mark_background(im,labels):
    """
    Mark segments with low intensity as background.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The multivariate image.
    labels : np.array, shape (M,N)
        The corresponding label map.

    Returns
    -------
    np.array, shape (M,N)
        The new label map.
    """
    bg_marked = labels.copy()
    stats = patch_stats(im,bg_marked)
    mean = stats['mean']
    area = stats['stats']['area']
    bg_labels = mean.loc[np.logical_or(mean.max(axis=1) < 10, np.logical_and(mean.max(axis=1) < 60, area > 100))].index
    bg_mask = np.isin(bg_marked,bg_labels)
    bg_marked[bg_mask] = BG_VAL
    return bg_marked


@profile
def cluster(im, labels, n=10, what=['mean','coords'], touch=False):
    """
    Merge labels based on a KMeans clustering.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The multivariate image.
    labels : np.array, shape (M,N)
        The corresponding label map.
    n : int, optional
        The number of clusters (default: 10)
    what : list, optional
        The properties of each segment to use for clustering. Must be a sublist of
        ['mean', 'var', 'coords', 'stats']. (default: ['mean','coords'])
    touch : bool, optional
        Whether only to merge segments if they physically touch (share an edge). (default: False)

    Returns
    -------
    np.array, shape (M,N)
        The new label map.
    """
    ## Clustering
    stats = patch_stats(im,labels,what=what)
    X = pd.concat([stats[_] for _ in what], axis=1)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clust = KMeans(n_clusters=n, random_state=0).fit(X_scaled)
    # clust = AgglomerativeClustering(n_clusters=100).fit(X_scaled)

    cl = clust.labels_
    x = X.index

    clustered = labels.copy()
    cl1,cl2 = np.meshgrid(cl, cl, copy=False)
    merge = pd.DataFrame(cl1==cl2, index=x, columns=x)
    clustered = merge_connected(clustered, merge, touch=touch)

    return clustered

    # ---------
    # X_spatial = coords
    # X_color = mean
    # clist = []
    # for X,n in [ (X_spatial,100), (X_color,5) ]:
    #     scaler = preprocessing.StandardScaler().fit(X)
    #     X_scaled = scaler.transform(X)
    #     clust = KMeans(n_clusters=n, random_state=0).fit(X_scaled)
    #     x = X.index
    #     cl = clust.labels_
    #     clustered = new_labels.copy()
    #     for l in np.unique(cl): clustered[np.isin(clustered,x[cl==l])] = l
    #     clist.append(clustered)
    #
    # clustered_joined = skseg.join_segmentations(clist[0],clist[1])
    # clustered_joined[new_labels == BG_VAL] = BG_VAL
    # len(np.unique(clustered_joined))
    # cv2.imwrite('sample_segments.jpg', colorize(clustered_joined,10))
    # cv2.imwrite('sample_segments.jpg', colorize(clist[1],5))



def colorize(labels,N=10):
    """
    Apply a color map to a map of integer labels.

    Parameters
    ----------
    labels : np.array, shape (M,N)
        The labeled image.
    N : int, optional
        The number of colors to use (default: 10)

    Returns
    -------
    np.array, shape (M,N,3)
        A colored image in BGR space, ready to be handled by OpenCV.
    """
    seg = (labels % N) * (255/(N-1))
    seg_gray = cv2.cvtColor(seg.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    seg_color = cv2.applyColorMap(seg_gray, cv2.COLORMAP_JET)
    seg_color[labels==BG_VAL] = 0
    return seg_color


def extract_first(im):
    """
    Extract index of first non-zero band for each pixel.

    NOTE: The result is indexed from one rather than zero, to distinguish
    a signal in the first band from the case where all bands are zero.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The multivariate image.

    Returns
    -------
    np.array, shape (M,N,1)
        The index of the first non-zero band (indexed from 1) for each pixel.
    """
    any_signal = im.sum(axis=2) > 0
    first_idx = im.argmax(axis=2).astype(np.uint8)
    result = np.expand_dims(first_idx, axis=-1) + 1
    result[~any_signal] = 0
    return result


@profile
def load_im(sub=False):
    ##
    ## Load sub image for development
    ##
    path = '/Users/jhansen/ED/ecometrica/bmap_a12886_mosaic__0-20160903_1-20161003_2-20161027_3-20161202_4-20161226_5-20170119_6-20170212_7-20170308_8-20170401_9-20170425_10-20170519_11-20170612_12-20170706_13-20170730_14-20170823.tif'
    src = gdal.Open(path)
    data = src.ReadAsArray()
    if sub:
        data = data[:,18000:20000,5500:7500]
    im = data.transpose((1,2,0))
    return im


@profile
def segment():
    """
    """
    im = load_im(sub=True)
    first = extract_first(im)
    blurred = cv2.GaussianBlur(im, ksize=(5,5), sigmaX=0.5)
    seg = skseg.felzenszwalb(blurred, scale=1.0, sigma=0, min_size=10, multichannel=True)
    seg += 1
    new_labels = seg.copy()
    new_labels = mark_background(im, new_labels)
    new_labels = bulk_merge(im, new_labels, 'distance', threshold=60, touch=True)
    new_labels = cluster(im, new_labels, n=10, what=['mean'], touch=True)
    new_labels = merge_small(im, new_labels, 100)

    new_labels_2 = bulk_merge(im, new_labels, 'ttest', threshold=1e-4, touch=True)
    new_labels_2 = bulk_merge(im, new_labels, 'distance', threshold=10, touch=False)
    # new_labels_2 = bulk_merge(im, new_labels, 'ttest', threshold=0.00001, touch=True)
    # new_labels_2 = bulk_merge(im, new_labels, 'shape', threshold=1000, touch=True)
    # new_labels_2 = bulk_merge(im, new_labels, 'edge', threshold=0.1, touch=True)
    # new_labels_2 = cluster(im, new_labels, n=100, what=['mean','coords'], touch=False)
    new_labels_2 = cluster(im, new_labels_2, n=10, what=['mean'], touch=False)
    # new_labels_2 = cluster(first, new_labels, n=5, what=['mean','var'], touch=False)
    # new_labels_2 = cluster(first, new_labels, n=100, what=['mean','var','coords'], touch=False)
    new_labels_2 = merge_small(im, new_labels_2, 100)

    len(np.unique(new_labels_2))
    cv2.imwrite('sample_segments.jpg', colorize(new_labels_2))
    # cv2.imwrite('sample_segments.jpg', scale_255(skseg.find_boundaries(new_labels_1)))

@profile
def profile_complete():
    im = load_im(sub=False)
    first = extract_first(im)
    blurred = cv2.GaussianBlur(first, ksize=(5,5), sigmaX=0.5)
    seg = layerwise_segments(blurred, threshold=30, bandwidth=50)


if __name__ == '__main__':
    ## DEBUG: For profiling
    segment()
