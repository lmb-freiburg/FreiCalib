import argparse, os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.general_util import json_load, fig2data

from calib_M import find_data, enumerate_points
from detect_marker import detect_marker
from core.BoardDetector import BoardDetector
from core.TagPoseEstimator import TagPoseEstimator
from core.EstimateM import calculate_reprojection_error, greedy_pick_object_pose, estimate_and_score_object_poses, calc_3d_object_points


def check_extrinsics(marker_path, data_path,
                     cam_pat, run_pat,
                     K, dist, M,
                     verbose):
    # find input data
    base_path, img_shapes, data, cam_ids = find_data(data_path, cam_pat, run_pat)

    # get detections
    det = list()
    for x in data:
        det.append(
            detect_marker(marker_path, x, output_file=None, cache=False, verbose=False)
        )

    # uniquely number detections
    detector = BoardDetector(marker_path)
    tagpose = TagPoseEstimator(detector.object_points)
    p2d, pid, p3d, fid, cid, mid = enumerate_points(det, detector.object_points)

    # calculate object poses and pick greedily
    scores_object, T_obj2cam = estimate_and_score_object_poses(tagpose, p2d, cid, fid, mid, K, dist)
    object_poses = greedy_pick_object_pose(scores_object, T_obj2cam, M, verbose)
    point3d_coord, pid2d_to_pid3d = calc_3d_object_points(tagpose.object_points, object_poses, fid, cid, mid)

    # find reprojection error
    err_mean, error_per_cam = calculate_reprojection_error(p2d, point3d_coord, pid2d_to_pid3d,
                                                           K, dist, M,
                                                           cid, return_cam_wise=True)

    # give some output of the results
    hist_per_cam = dict()
    print('Error per cam:')
    for cid, error_cam in error_per_cam.items():
        y, x = np.histogram(error_cam)
        x = 0.5*(x[:-1] + x[1:])
        hist_per_cam[cid] = (x, y)
        print('\t> Cam%d: %.3f (px)' % (cid, np.mean(error_cam)))
    print('Mean error: %.2f pixels' % err_mean)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for cid, (x, y) in hist_per_cam.items():
        ax.plot(x, y, label='cam%d' % cid)
    plt.legend()
    plt.title('Reprojection error distribution')
    plt.xlabel('Pixels')
    plt.xlabel('Occurences')
    img = fig2data(fig)
    cv2.imshow('stats', img[:, :, ::-1])
    cv2.waitKey()


def load_calib(calib_path):
    # load file
    calib = json_load(calib_path)

    # find id's
    cid_list = list()
    for cid in range(1024):
        if 'cam%d' % cid in calib['K'].keys():
            cid_list.append(cid)

    # bring in different layout
    calib_out = {'K': list(),
                 'dist': list(),
                 'M': list()}

    for cid in cid_list:
        calib_out['K'].append(np.array(calib['K']['cam%d' % cid]))
        calib_out['dist'].append(np.array(calib['dist']['cam%d' % cid]))
        calib_out['M'].append(np.array(calib['M']['cam%d' % cid]))

    return calib_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check extrinsic calibration wrt some data.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data is.')
    parser.add_argument('calib_path', type=str, help='Path to where the calibration data is.')
    parser.add_argument('--cam_pat', type=str, default='cam%d', help='Expression that is being matched '
                                                                     'when searching for camera folders or names.')
    parser.add_argument('--run_pat', type=str, default='run%03d', help='Expression that is being matched '
                                                                       'when searching for runs.')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level, higher is more ouput.')
    args = parser.parse_args()

    # load existing calib data
    calib = load_calib(args.calib_path)

    check_extrinsics(args.marker, args.data_path,
                     args.cam_pat, args.run_pat,
                     calib['K'], calib['dist'], calib['M'],
                     args.verbosity)
