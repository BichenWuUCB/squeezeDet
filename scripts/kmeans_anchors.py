# Author: Mihai Martalogu (mihai@martalogu.com) 06/16/2017
"""
Computes anchor sizes optimized for your dataset
"""
import os
import subprocess
import itertools
import argparse
import inspect
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm
import concurrent.futures



def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))


def main():
    parser = argparse.ArgumentParser(description='Computes anchor sizes optimized for your dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-layout', default='kitti', choices=['kitti'],
                        help='Layout of the dataset')
    parser.add_argument('-j', '--jobs', default=multiprocessing.cpu_count() * 2,
                        help='Number of parallel jobs to spawn')
    
    script_path = inspect.stack()[0][1]     # https://stackoverflow.com/a/6628348
    parser.add_argument('--dataset-root',
                        default=os.path.abspath(os.path.join(os.path.dirname(script_path),
                                                             '../data/KITTI')),
                        help='Root directory of the dataset')
    parser.add_argument('--geometry', default='320x427',
                        help='Geometry of the image input for the neural net, in the form of a' +
                             ' <width>x<height> string')
    parser.add_argument('--k', default=9, type=int,
                        help='Number of anchors')
    parser.add_argument('--kmeans-max-iter', default=1000, type=int,
                        help='Maximum number of iterations of the k-means algorithm')
    args = parser.parse_args()
    input_w, input_h = (int(x) for x in args.geometry.split('x'))
    metadata = get_dataset_metadata(args.dataset_root, input_w, input_h, args.jobs)
    print_anchors(args.k, metadata, max_iter=args.kmeans_max_iter)


def get_dataset_metadata(dataset_root, input_w, input_h, max_jobs):
    """
    Load all dataset metadata into memory. You might need to adapt this if your dataset is really huge.
    """
    nonlocals = {   # Python 2 doesn't support nonlocal, using a mutable dict() instead
        'entries_done': 0,
        'metadata': dict(),
        'entries_done_pbar': None
    }
    with open(os.path.join(dataset_root, 'ImageSets', 'trainval.txt')) as f:
        dataset_entries = f.read().splitlines()
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_jobs) as pool:

        for entry in tqdm(dataset_entries, desc='Scheduling jobs'):
            if nonlocals['entries_done_pbar'] is None:
                # instantiating here so that it appears after the 'Scheduling jobs' one
                nonlocals['entries_done_pbar'] = tqdm(total=len(dataset_entries), desc='Retrieving metadata')

            def entry_done(future):
                """ Record progress """
                nonlocals['entries_done'] += 1
                nonlocals['entries_done_pbar'].update(1)
                fr = future.result()
                if fr is not None:
                    local_entry, value = fr     # do NOT use the entry variable from the scope!
                    nonlocals['metadata'][local_entry] = value

            future = pool.submit(get_entry_metadata, dataset_root, entry, input_w, input_h)
            future.add_done_callback(entry_done)      # FIXME: doesn't work if chained directly to submit(). bug in futures? reproduce and submit report.
    nonlocals['entries_done_pbar'].close()
    assert len(nonlocals['metadata'].values()) >= 0.9 * len(dataset_entries)    # catch if entry_done doesn't update the dict correctly
    return nonlocals['metadata']


def get_entry_metadata(dataset_root, entry, input_w, input_h):
    """
    NOTE: this must be a global function (so that it can be used by ProcessPoolExecutor)
    """
    id_cmd = ['gm', 'identify', '-format', '%G', os.path.join(dataset_root, 'training', 'image_2', entry + '.*')]
    try:
        out = subprocess.check_output(id_cmd)
    except subprocess.CalledProcessError:
        print 'Unable to process entry', entry, '(ignoring)'
        return None

    img_w, img_h = [int(x) for x in out.split('x')]
    scaleX = float(input_w) / img_w
    scaleY = float(input_h) / img_h

    bbox_sizes = []
    with open(os.path.join(dataset_root, 'training', 'label_2', entry + '.txt')) as f:
        for line in f.read().splitlines():
            if line:
                # n02676566 0.0 0 0 21 1 376 547 -1 -1 -1 -1 -1 -1 -1
                xmin, ymin, xmax, ymax = [float(x) for x in line.split()[4:8]]
                w = xmax - xmin
                h = ymax - ymin
                bbox_sizes.append((w * scaleX, h * scaleY))

    return (entry, {
        'size': (img_w, img_h),
        'bbox_sizes': bbox_sizes
    })


def print_anchors(k, metadata, max_iter):
    bbox_sizes = np.array(flatten([m['bbox_sizes'] for m in metadata.values()]))
    plt.scatter(bbox_sizes[:, 0], bbox_sizes[:, 1])
    plt.xlabel('width')
    plt.ylabel('height')

    ARBITRARY_SEED = 13

    kmeans = KMeans(n_clusters=k, random_state=ARBITRARY_SEED, max_iter=max_iter).fit(bbox_sizes)
    centroids = kmeans.cluster_centers_
    print_nicely(centroids)

    plt.gca().set_aspect('equal', adjustable='box')

    plt.plot(centroids[:, 0], centroids[:, 1], 'x', color='red')

    plt.show()


def print_nicely(centroids):
    out = '\n\n# Replace the Python list in anchor_shapes (in set_anchors() under src/config/kitti_<your_model>_config.py) with this one:\n'
    out += '['
    centroids_list = centroids.tolist()
    for idx, entry in enumerate(centroids_list):
        out += '[%.2f, %.2f]' % (entry[0], entry[1])
        comma, space, newline = '', '', ''
        if idx != len(centroids_list) - 1:
            comma = ','
            space = ' '
            if idx % 3 == 2:
                newline = '\n'
                space = ''
        out += comma + space + newline
    out += ']'
    print out



if __name__=='__main__':
    main()