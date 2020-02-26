import argparse, os
import numpy as np

from core.BoardDetector import BoardDetector
from utils.general_util import json_load, json_dump

from detect_marker import detect_marker
from core.EstimateK import estimate_intrinsics


def enumerate_points(det, p2d, pid, max_pts):
    """ Flattens the list of detections and assigns unique ids for points. """
    p2d_out, pid_out, p3dm_out, fid_out = list(), list(), list(), list(),

    # iterate frames
    for fid, (this_p2d, this_pid) in enumerate(zip(p2d, pid)):
        assert max_pts >= len(this_p2d), 'Detected more keypoints in one frame than this detector can have!'
        p2d_out.extend(this_p2d)   # image coordinates for this detected point
        pid_out.extend([fid*max_pts + i for i in this_pid])   # unique point id (over all frames we have observed)
        p3dm_out.extend(det.get_matching_objectpoints(this_pid))  # 3D location in the model frame
        fid_out.extend([fid for _ in this_pid])  # frame id of this point

    p2d_out = np.array(p2d_out)
    pid_out = np.array(pid_out)
    p3dm_out = np.array(p3dm_out)
    fid_out = np.array(fid_out)
    return p2d_out, pid_out, p3dm_out, fid_out


def calc_intrinsics(marker_path, data_path, det_file_name, output_file=None,
                    estimate_dist=True, dist_complexity=5,
                    cache=False, verbose=0):
    if os.path.isdir(data_path):
        base_dir = data_path
    else:
        base_dir = os.path.dirname(data_path)

    # try to load precomputed
    if output_file is not None:
        calib_file = os.path.join(base_dir, output_file)
        if cache and os.path.exists(calib_file):
            if verbose > 0:
                print('Loading intrinsic calibration from: %s' % calib_file)
            calib = json_load(calib_file)
            return np.array(calib['K']), np.array(calib['dist'])

    if verbose > 0:
        print('Calculating intrinsic calibration for:')
        print('\tData path: %s' % data_path)
        print('\tMarker file: %s' % marker_path)

    # set up detector and estimator
    detector = BoardDetector(marker_path)

    if os.path.isdir(data_path):
        if verbose > 0:
            print('\tAssuming: Folder of images.')
        base_dir = data_path
    else:
        if verbose > 0:
            print('\tAssuming: Video file.')
        base_dir = os.path.dirname(data_path)

    # check for detections
    if det_file_name is None:
        det = detect_marker(marker_path, data_path, cache=cache, verbose=verbose - 1)
    else:
        detections_file = os.path.join(base_dir, det_file_name)

        if not os.path.exists(detections_file):
            if verbose > 1:
                print('Could not locate marker detections. Running detector now and saving them to folder.')
            det = detect_marker(marker_path, data_path, det_file_name, verbose=verbose-1)

        else:
            det = json_load(detections_file)

    # give points unique ids
    max_num_pts = len(detector.object_points)
    p2d, pid, p3dm, fid = enumerate_points(detector, det['p2d'], det['pid'], max_num_pts)
    if verbose > 0:
        print('Found %d unique points to estimate intrinsics from.' % pid.shape[0])

    # estimate intrinsics
    K, dist = estimate_intrinsics(p2d, fid, p3dm, det['img_shape'],
                                  estimate_dist=estimate_dist,
                                  dist_complexity=dist_complexity,
                                  verbose=verbose)

    # save intrinsics
    if output_file is not None:
        calib = {'K': K, 'dist': dist}
        json_dump(calib_file, calib, verbose=verbose > 0)
    return K, dist


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate intrinsic calibration.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data is.')
    parser.add_argument('-c', '--cache', action='store_true', help='Use stored version.')
    parser.add_argument('--estimate_dist', action='store_true', help='Estimate distortion.')
    parser.add_argument('--dist_complexity', type=int, default=2, help='How many distortion parameters to estimate.'
                                                                       ' Should be in [0, 3]')
    parser.add_argument('--det_file_name', type=str, default='detections.json',
                        help='File to store detections in.')
    parser.add_argument('--calib_file_name', type=str, default='K.json',
                        help='File to store calibration result in.')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level, higher is more ouput.')
    args = parser.parse_args()

    calc_intrinsics(args.marker, args.data_path, args.det_file_name,
                    args.calib_file_name,
                    args.estimate_dist, args.dist_complexity,
                    args.cache,
                    args.verbosity)
