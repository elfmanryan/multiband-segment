# import bunge
from pathlib import Path
#import sys
#import importlib
#import cv2
#import rasterio
import numpy as np
#import pandas as pd
import segmentation as seg



bmap_str = r"~/2b3353e10c93e93fed916ecbbf157590_rel_orb_26_bmap.tif"
bmap_path = Path(bmap_str)
save_tif_path = bmap_path.parent / (bmap_path.stem + 'segment_.tif')


im = seg.load_im(bmap_str)

labels = seg.segment(im,
                     first=False, #only use pixels of first change to segment
                     sigma=0.6, #level of blurring
                     start_time_index=0, #the default index before which change is ignored
                     min_size=5, #segments below this sie are merged with neighbours
                     pmin=0.05, #t-test with whith to merge neighbours
                     dmax=20, #he Euclidean feature distance threshold for merging segments
                     nclusters=5, #A KMeans clustering will be performed
                     # with k = `nclusters` (default: 0). Only segments that belong to the
                     # same cluster _and_ are adjacent to each other will be merged. `0` means no clustering.
                     ngroups=0)

#cast to correct data format
labels = labels.astype(np.int32) #recast to correct dtype


#extract frequency to get statistics later
#frequency_change = seg.extract_frequency(im, pixel_value=255)
first_change = seg.extract_first(im, reduce=True)

#get statistics from array  (e.g. firt change) for labelled patches
stats = seg.patch_stats(first_change, labels)

#extract means values
means = stats['mean']# get the mean of frequency

#reapply statistics to labels to make a new map of statistics
map_time_first_change = np.vectorize(lambda i: means.iloc[i])
map_time_first_change(labels)
