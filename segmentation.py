import sys 
from osgeo import gdal
import numpy as np
import cv2
import scipy.stats
from scipy.sparse import csgraph, csr_matrix
import pandas as pd
from sklearn.cluster import KMeans
from skimage import segmentation as skseg
from skimage.measure import regionprops
from sklearn import preprocessing

BG_VAL = -1

try:
    type(profile)
except NameError:
    def profile(fn): return fn


@profile
def neighbor_matrix(labels, bg=True, connectivity=4, touch=True):
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
        np.array([[1, 0, 0]]),
        np.array([[0, 0, 1]]),
        np.array([[1, 0, 0]]).T,
        np.array([[0, 0, 1]]).T
    ]
    if connectivity==8:
        kernels.extend([
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        ])
    shifted = np.stack([cv2.filter2D(labels.astype(np.float32), -1, k) for k in kernels], axis=-1)
    if touch:
        bg_neighbors = shifted[labels != BG_VAL]
    else:
        bg_neighbors = shifted[labels == BG_VAL]

    bg_neighbors[bg_neighbors == BG_VAL] = np.nan
    bg_neighbors = bg_neighbors[~np.isnan(bg_neighbors).all(axis=1)]
    _mins = np.nanmin(bg_neighbors, axis=1).astype(np.int32)
    _maxs = np.nanmax(bg_neighbors, axis=1).astype(np.int32)
    npairs = np.stack([_mins, _maxs], axis=-1)[_mins != _maxs]
    idx = np.arange(len(x))
    lookup = dict(np.stack([x, idx], axis=-1))
    npairs_idx = np.vectorize(lambda x: lookup[x], otypes=[np.int32])(npairs)
    result = np.zeros((len(x),)*2, dtype=bool)
    result[npairs_idx[:, 0], npairs_idx[:, 1]] = True
    result[npairs_idx[:, 1], npairs_idx[:, 0]] = True
    result[x==BG_VAL, :] = False
    result[:, x==BG_VAL] = False
    # DEBUG: Somehow this line is very expensive:
    # result = np.logical_or(result, result.T)
    return pd.DataFrame(result, index=x, columns=x)


@profile
def edge_length(labels, bg=True, connectivity=4):
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
        np.array([[1, 0, 0]]),
        np.array([[0, 0, 1]]),
        np.array([[1, 0, 0]]).T,
        np.array([[0, 0, 1]]).T
    ]
    if connectivity == 8:
        kernel.extend([
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]]),
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]),
        ])
    shifted = np.stack([cv2.filter2D(labels.astype(np.float32), -1, k) for k in kernels], axis=-1)
    if not bg: shifted[shifted == BG_VAL] = np.nan
    _mins = np.nanmin(shifted, axis=2).astype(np.int32)
    _maxs = np.nanmax(shifted, axis=2).astype(np.int32)
    edge = (_mins != _maxs)
    l1 = _mins[edge]
    l2 = _maxs[edge]
    pairs = np.stack([l1, l2], axis=-1)
    pairs_idx = _replace_labels(pairs, pd.Series(np.arange(len(x)), index=x))
    result = np.zeros(x.shape*2, dtype=np.int32)
    result[pairs_idx[:, 0],pairs_idx[:, 1]] += 1
    result[pairs_idx[:, 1],pairs_idx[:, 0]] += 1
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
        fill_index = np.arange(labels.min(), labels.max() + 1)
        replace = pd.Series(fill_index, index=fill_index).copy()
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
        nm = neighbor_matrix(labels, touch=True, bg=False)
        merge = np.logical_and(connected, nm)
    else:
        merge = connected

    csr = csr_matrix(merge)
    cc = csgraph.connected_components(csr, directed=False)
    replace = pd.Series(cc[1], index=x)
    result = _remap_labels(labels, replace)
    return result


@profile
def bulk_merge(im, labels, mode='distance', threshold=None, touch=True):
    """
    Compute segment distance matrix and merge similar segments.
    """
    if touch:
        nm = neighbor_matrix(labels, touch=True, bg=False)
    else:
        ll = np.unique(labels)
        ll = ll[ll != BG_VAL]
        nm = pd.DataFrame(np.ones((len(ll), len(ll)), dtype=bool), index=ll, columns=ll)

    # ------------------------------------------------------------------------
    if mode == 'distance':
        # distance-based
        stats = patch_stats(im, labels, what=['mean'])
        mean = stats['mean']
        x = mean.index
        if threshold is None: threshold = 50
        # dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(stats['mean'], 'euclidean'))
        # similar = (dist < threshold)
        i1,i2 = np.meshgrid(x, x, copy=False)
        p1 = mean.loc[i1[nm]]
        p2 = mean.loc[i2[nm]]
        d = np.linalg.norm(p1.values - p2.values, axis=1)
        similar = np.zeros(nm.shape, dtype=bool)
        similar[nm] = (d < threshold)

    elif mode == 'ttest':
        # t-test based
        stats = patch_stats(im, labels, what=['mean', 'var', 'stats'])
        mean = stats['mean']
        var = stats['var']
        area = stats['stats']['area']
        x = mean.index
        if threshold is None: threshold = 0.0001

        eps_var = 0 # Avoid zero variance.
        x1, x2 = np.meshgrid(x, x, copy=False)
        i1 = x1[nm]
        i2 = x2[nm]
        x1_mean = mean.loc[i1].values
        x2_mean = mean.loc[i2].values
        x1_var = var.loc[i1].values + eps_var
        x2_var = var.loc[i2].values + eps_var
        x1_size = np.expand_dims(area.loc[i1].values, axis=-1)
        x2_size = np.expand_dims(area.loc[i2].values, axis=-1)
        t = np.abs(x1_mean - x2_mean) / np.sqrt(x1_var/x1_size + x2_var/x2_size)
        df = (x1_var/x1_size + x2_var/x2_size)**2 / (x1_var**2 / (x1_size**2 * (x1_size-1)) + x2_var**2 / (x2_size**2 * (x2_size-1)))
        p = 1 - (scipy.stats.t.cdf(t, df) - scipy.stats.t.cdf(-t, df))
        p_ = np.nan_to_num(p).min(axis=1)
        similar_flat = (p_ > threshold)
        similar = np.zeros(nm.shape, dtype=bool)
        similar[nm] = similar_flat

    elif mode == 'shape':
        if threshold is None: threshold = 100
        stats = patch_stats(im, labels, what=['stats', 'coords'])
        area = stats['stats']['area']
        coords = stats['coords']
        x = area.index
        i1, i2 = np.meshgrid(x, x, copy=False)
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
        e = edge_length(labels, bg=False)
        stats = patch_stats(im, labels, what=['stats'])
        # area = stats['stats']['area']
        peri = stats['stats']['perimeter']
        x = peri.index
        i1, i2 = np.meshgrid(x, x, copy=False)
        # area1 = area.loc[i1[nm]]
        # area2 = area.loc[i2[nm]]
        peri1 = peri.loc[i1[nm]]
        peri2 = peri.loc[i2[nm]]
        edges = e.values[nm]
        score = edges / (peri1.values + peri2.values)
        similar = np.zeros(nm.shape, dtype=bool)
        similar[nm] = (score > threshold)

    elif mode == 'kmeans':
        stats = patch_stats(im, labels, what=['mean', 'coords'])
        x = stats['mean'].index
        mean = stats['mean']
        # var = stats['var']
        # yxgrid = np.stack(np.mgrid[0:im.shape[1],0:im.shape[0]], axis=2)
        # coords = patch_stats(yxgrid,labels)['mean']
        # coords.columns = ['y', 'x']
        # X = pd.concat([mean,coords], axis=1)
        X = mean
        # X = pd.concat([mean, var], axis=1)
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        clust = KMeans(n_clusters=threshold, random_state=0).fit(X_scaled)
        cl = clust.labels_
        l1, l2 = np.meshgrid(cl, cl, copy=False)
        similar = (l1 == l2)

    np.fill_diagonal(similar, False)

    # ------------------------------------------------------------------------
    if touch: merge = np.logical_and(similar, nm)
    merged_labels = merge_connected(labels, merge)
    # ------------------------------------------------------------------------
    # merged_labels = relabel(merged_labels)
    return merged_labels

    # ------------------------------------------------------------------------
    cv2.imwrite('sample_segments.jpg', colorize(merged_labels))
    # ------------------------------------------------------------------------


    # multivariate t-test?
    # cov = stats['cov']
    x1_cov, x2_cov = np.meshgrid(stats['cov'], stats['cov'], copy=False)
    cov = (x1_size-1) * x1_cov + (x2_size-1) * x2_cov
    d_mu = x1_mean - x2_mean
    Tsq = x1_size * x2_size / (x1_size + x2_size) * d_mu * cov * d_mu
    # ...
    from spm1d.stats import hotellings2
    _data2 = pd.Series(index=ll, dtype=np.object)
    for l, r in _data.iterrows():
        _data2.loc[l] = np.stack(r, axis=-1)
    for l, r in _data2.iteritems():
        _data2.loc[l] = r.T
    T2 = hotellings2(_data2.loc[2], _data2.loc[3])
    _data3 = np.expand_dims(_data2, axis=-1)
    def hdist(u, v):
        try:
            return hotellings2(u[0], v[0]).inference(0.01).h0reject
        except:
            return np.nan

    T2_grid = scipy.spatial.distance.pdist(_data3, hdist)


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
    stats = patch_stats(im, new_labels)
    mean = stats['mean']
    area = stats['stats']['area']
    small = (area < threshold)
    nm = neighbor_matrix(new_labels, touch=True, bg=False)
    _nm = nm.loc[small, :]
    _i1, _i2 = np.meshgrid(mean.loc[:].index, mean.loc[small].index, copy=False)
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

    return new_labels


@profile
def patch_stats(im, labels, what=['mean', 'stats']):
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
        ['mean', 'var', 'cov', 'coords', 'stats']. (default: ['mean', 'coords'])

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
    try:
        offset = labels[labels != BG_VAL].min() - 1
    except ValueError:
        offset = 0
    _labels = labels - offset
    result = {}
    if 'mean' in what:
        result['mean'] = pd.DataFrame(index=ll, columns=np.arange(ndim))
    if 'var' in what:
        result['var'] = pd.DataFrame(index=ll, columns=np.arange(ndim))
    if 'cov' in what:
        result['cov'] = pd.Series(index=ll, dtype=np.object)
        _data = pd.DataFrame(index=ll, columns=np.arange(ndim))

    for i in range(ndim):
        props = regionprops(_labels, _im[:, :, i])
        if 'mean' in what:
            result['mean'][i] = np.array([_.mean_intensity for _ in props])
        if 'var' in what:
            result['var'][i] = np.array([np.var(_.intensity_image[_.image]) for _ in props])
        if 'cov' in what:
            _data[i] = np.array([_.intensity_image[_.image] for _ in props])

    if 'cov' in what:
        for l, r in _data.iterrows(): result['cov'].loc[l] = np.cov(np.stack(r))

    if 'stats' in what:
        result['stats'] = pd.DataFrame(index=ll)
        result['stats']['area'] = np.array([_.area for _ in props])
        result['stats']['perimeter'] = np.array([_.perimeter for _ in props])

    if 'coords' in what:
        result['coords'] = pd.DataFrame(np.array([_.centroid for _ in props]), index=ll, columns=['x', 'y'])

    return result


@profile
def mark_background(im, labels):
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
    stats = patch_stats(im, bg_marked)
    mean = stats['mean']
    area = stats['stats']['area']
    limit = im.max()
    min_val_all = limit * 0.04
    min_val_large_area = limit * 0.25
    bg_labels = mean.loc[np.logical_or(mean.max(axis=1) < min_val_all, np.logical_and(mean.max(axis=1) < min_val_large_area, area > 100))].index
    bg_mask = np.isin(bg_marked, bg_labels)
    bg_marked[bg_mask] = BG_VAL
    return bg_marked


@profile
def cluster(im, labels, n=10, what=['mean', 'coords'], touch=False):
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
        ['mean', 'var', 'coords', 'stats']. (default: ['mean', 'coords'])
    touch : bool, optional
        Whether only to merge segments if they physically touch (share an edge). (default: False)

    Returns
    -------
    np.array, shape (M,N)
        The new label map.
    """
    ## Clustering
    what_with_area = what
    if 'stats' not in what_with_area:
        what_with_area += ['stats']
    stats = patch_stats(im, labels, what=what_with_area)
    weights = stats['stats']['area']
    X = pd.concat([stats[_] for _ in what], axis=1)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    clust = KMeans(n_clusters=n, random_state=0).fit(X_scaled,
                                                     sample_weight=weights)

    cl = clust.labels_
    x = X.index

    clustered = labels.copy()
    cl1, cl2 = np.meshgrid(cl, cl, copy=False)
    merge = pd.DataFrame(cl1==cl2, index=x, columns=x)
    clustered = merge_connected(clustered, merge, touch=touch)

    return clustered


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


def extract_first(im, reduce=True):
    """
    Extract index of first non-zero band for each pixel.

    NOTE: The result is indexed from one rather than zero, to distinguish
    a signal in the first band from the case where all bands are zero.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The multivariate image.
    reduce : bool, optional
        If True, reduce image to shape (M,N,1) where each value is the index of first change (default: True).
        Otherwise, retain array structure but remove all changes after the first.

    Returns
    -------
    np.array, shape (M,N,C) or (M,N,1)
        The index of the first non-zero band (indexed from 1) for each pixel.
    """
    any_signal = im.sum(axis=2) > 0
    first_idx = im.argmax(axis=2).astype(np.uint8)
    if reduce:
        result = np.expand_dims(first_idx, axis=-1) + 1
        result[~any_signal] = 0
    else:
        result = np.zeros_like(im)
        result[any_signal, first_idx[any_signal]] = 255
    return result


def segment(im, init='felzenszwalb', first=True, sigma=0, start_time_index=0, min_size=100, pmin=0, dmax=0, nclusters=0, ngroups=0):
    """
    Segment an image.

    Parameters
    ----------
    im : np.array, shape (M,N,C)
        The image to segment.
    init : str, optional
        The segment initiation method. Currently only supports `felzenswalb`
        (default: 'felzenszwalb').
    first : bool, optional
        Whether to only use the time of first change for the segmentation (default: True).
    sigma : float, optional
        Blur the data before segmenting using a Gaussian filter with parameter sigma.
        `0` means no blurring (default: 0).
    start_time_index : int, optional
        The index of the time before which to ignore all changes (default: 0).
    min_size : int, optional
        Segments below this threshold will be merged with a neighboring segment (default: 100).
    pmin : float, optional
        The threshold for the p-value of the t-test below which segments will be merged (default: 0).
        `0` means no t-test based merging.
    dmax : float, optional
        The Euclidean feature distance threshold for merging segments (default: 0).
        `0` means no distance based merging.
    nclusters : int, optional
        A KMeans clustering will be performed with k = `nclusters` (default: 0). Only segments
        that belong to the same cluster _and_ are adjacent to each other will be merged.
        `0` means no clustering.
    ngroups : int, optional
        Return exactly `ngroups` different segments which will not need to touch.
        `0` means no such grouping will be done (default: 0).

    Returns
    -------
    np.array, shape (M,N)
        An MxN array, where each pixel contains an integer index of the segment 
        it belongs to. Identified background pixels are labeled with `-1`.
    """
    data = im.copy()
    if start_time_index > 0:
        data[:, :, :start_time_index] = 0
    if first:
        # Extract time of first change
        data = extract_first(data, reduce=False)
        what = ['mean', 'var']
    else:
        what = ['mean']
    if sigma > 0:
        # Do a Gaussian blur of the data
        data = cv2.GaussianBlur(data, ksize=(5, 5), sigmaX=sigma)
    # Initiate segments using Felzenszwalb algorithm
    if init == 'felzenszwalb':
        labels = skseg.felzenszwalb(data, scale=0.9, sigma=0, min_size=20, multichannel=True)
        labels += 1
    else:
        raise ValueError('"%s" is not a valid initiation method.' % init)
    # Mask out detected background
    labels = mark_background(data, labels)
    # Merge segments based on pairwise similarity (distance)
    if dmax > 0:
        labels = bulk_merge(data, labels, 'distance', threshold=dmax, touch=True)
    # Merge segments based on pairwise similarity (ttest)
    if pmin > 0:
        labels = bulk_merge(data, labels, 'ttest', threshold=pmin, touch=True)
    # Merge segments based on KMeans clustering
    if nclusters > 0:
        labels = cluster(data, labels, n=nclusters, what=what, touch=True)
    # Merge small segments
    if min_size > 0:
        labels = merge_small(data, labels, min_size)
    # Perform a final clustering, merging also non-touching segments
    if ngroups > 0:
        labels = cluster(data, labels, n=ngroups, what=what, touch=False)

    return labels


@profile
def load_im(path, sub=False):
    ##
    ## Load sub image for development
    ##
    src = gdal.Open(path)
    data = src.ReadAsArray()
    if sub:
        data = data[:, 18000:20000, 5500:7500]
    im = data.transpose((1, 2, 0))
    return im


if __name__ == '__main__':
    im = load_im(path=sys.argv[1], sub=True)
    labels = segment(im, first=True, sigma=0.6, start_time_index=0, min_size=100, nclusters=5, ngroups=0)
    first = extract_first(im, reduce=True)
    cv2.imwrite('test_first_change.jpg', colorize(first[:, :, 0].astype(int) - 1, N=first.max()))
    cv2.imwrite('test_segmentation.jpg', colorize(labels, N=10))