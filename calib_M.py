import argparse, os
import numpy as np
import random
import cv2

from utils.general_util import find_images, sample_uniform, try_to_match, json_dump

from core.BoardDetector import BoardDetector
from core.TagPoseEstimator import TagPoseEstimator
from core.EstimateM import estimate_extrinsics_pnp, calculate_reprojection_error, run_bundle_adjust_pnp

from detect_marker import detect_marker
from calib_K import calc_intrinsics


def _check_number_of_frames(img_list):
    """ Checks if each entry of the list has the same length. """
    n = None
    for x in img_list:
        if n is None:
            n = len(x)
            assert n > 0, 'No image files found in folder.'

        assert n == len(x), 'Number of image frames differs between cameras.'


def _get_shape(x):
    """ Given a path to an image returns it shape as (H, W). """
    I = cv2.imread(x)
    assert I is not None, 'Reading image failed.'
    return I.shape[:2]


def _get_img_shapes(img_list):
    """ Checks if all images given by the list of paths have the same shape. """
    shape = None
    for x in sample_uniform(img_list, min(10, len(img_list))):
        if shape is None:
            shape = _get_shape(x)

        assert shape == _get_shape(x), 'Image shape changes between images of a single camera.'

    for x in random.sample(img_list, min(10, len(img_list))):
        assert shape == _get_shape(x), 'Image shape changes between images of a single camera.'

    return shape


def _get_shape_vid(x):
    """ Given a path to an video returns it shape as (H, W). """
    vid = cv2.VideoCapture(x)
    h = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    return int(h), int(w)


def find_data(data_path, cam_pat, run_pat, max_num=100):
    if os.path.isdir(data_path):
        # 1. Check for camera folders
        cam_folders, cam_ids = list(), list()
        for cid in range(max_num):
            p = os.path.join(data_path, cam_pat % cid)
            if os.path.exists(p):
                cam_folders.append(p)
                cam_ids.append(cid)

        assert len(cam_folders) > 0, 'No camera subfolder found.'

        # Parse folders of images
        img_list = [find_images(c) for c in cam_folders]

        # sanity check
        _check_number_of_frames(img_list)

        # get and check image shapes
        img_shapes = [_get_img_shapes(x) for x in img_list]

        return data_path, img_shapes, cam_folders, cam_ids

    else:
        # 2. Check for video files (then data_path points to one of the videos)
        assert os.path.isfile(data_path), 'A folder of cam%s folders or a video file should be given.'
        base_path = os.path.dirname(data_path)
        file_name = os.path.basename(data_path)
        cid = try_to_match(file_name, cam_pat, max_num)
        rid = try_to_match(file_name, run_pat, max_num)
        assert (cid is not None) and (rid is not None), 'Extracting patterns from video file failed.'

        video_files = [(i, file_name.replace(cam_pat % cid, cam_pat % i)) for i in range(max_num)]
        video_files = [(x[0], os.path.join(base_path, x[1])) for x in video_files]
        video_files = [x for x in video_files if os.path.exists(x[1])]
        cam_ids, video_files = zip(*video_files)

        vid_shapes = [_get_shape_vid(x) for x in video_files]

        return base_path, vid_shapes, video_files, cam_ids


def enumerate_points(det, model_points):
    """ Flattens the list of detections and assigns unique ids for points. """
    p2d_out, pid_out, p3dm_out, fid_out, cid_out, mid_out = list(), list(), list(), list(), list(), list()

    # iterate frames
    max_pts = model_points.shape[0]
    for cid, det_cam in enumerate(det):
        # iterate frames
        for fid, (this_p2d, this_pid) in enumerate(zip(det_cam['p2d'], det_cam['pid'])):
            assert max_pts >= len(this_p2d), 'Detected more keypoints in one frame than this detector can have!'
            p2d_out.extend(this_p2d)   # image coordinates for this detected point
            mid_out.extend(this_pid) # marker unique point id (same over frames)
            pid_out.extend([fid*max_pts + i for i in this_pid])   # unique point id (over all frames we have observed)
            p3dm_out.extend(model_points[this_pid])  # 3D location in the model frame
            fid_out.extend([fid for _ in this_pid])  # frame id of this point
            cid_out.extend([cid for _ in this_pid])  # camera id of this point

    p2d_out = np.array(p2d_out)
    pid_out = np.array(pid_out)
    p3dm_out = np.array(p3dm_out)
    fid_out = np.array(fid_out)
    cid_out = np.array(cid_out)
    mid_out = np.array(mid_out)
    return p2d_out, pid_out, p3dm_out, fid_out, cid_out, mid_out


def calc_extrinsics(marker_path, data_path, cam_pat, run_pat,
                    det_file_name, calib_file_name, calib_out_file_name,
                    estimate_dist, dist_complexity,
                    cache, verbose,
                    optimize_distortion=False, optimize_intrinsic=True):
    # find input data
    base_path, img_shapes, data, cam_ids = find_data(data_path, cam_pat, run_pat)

    # get detections
    det = list()
    for x, c in zip(data, cam_ids):
        det.append(
            detect_marker(marker_path, x,
                          det_file_name % c if det_file_name is not None else None,
                          cache=cache, verbose=verbose)
        )

    # uniquely number detections
    detector = BoardDetector(marker_path)
    tagpose = TagPoseEstimator(detector.object_points)
    p2d, pid, p3d, fid, cid, mid = enumerate_points(det, detector.object_points)

    # load/calc intrinsics for all cams
    K_list, d_list = zip(*[calc_intrinsics(marker_path, x,
                                           det_file_name % c if det_file_name is not None else None,
                                           calib_file_name % c if calib_file_name is not None else None,
                                           estimate_dist=estimate_dist, dist_complexity=dist_complexity,
                                           cache=cache, verbose=verbose) for c, x in zip(cam_ids, data)])
    K_list, d_list = np.array(K_list), np.array(d_list)

    # estimate extrinsic calibration
    M_list, point3d_coord, pid2d_to_pid3d, object_poses = estimate_extrinsics_pnp(tagpose, K_list, d_list,
                                                                                  p2d, cid, fid, pid, mid,
                                                                                  verbose)

    # calculate reprojection error of initial solution
    if verbose > 0:
        error = calculate_reprojection_error(p2d, point3d_coord, pid2d_to_pid3d,
                                             K_list, d_list, M_list,
                                             cid)
        print('Average reprojection error: %.2f pixels' % error)

    # run bundle adjust
    K_list, d_list, M_list, \
    point3d_coord = run_bundle_adjust_pnp(K_list, d_list, M_list,
                                          p2d, cid, fid, mid,
                                          detector.object_points, object_poses, img_shapes,
                                          optimize_intrinsic=optimize_intrinsic,
                                          optimize_distortion=optimize_distortion,
                                          verbose=verbose)

    # calculate reprojection error of the new solution
    if verbose > 0:
        error = calculate_reprojection_error(p2d, point3d_coord, pid2d_to_pid3d,
                                             K_list, d_list, M_list,
                                             cid)
        print('Average reprojection error: %.2f pixels' % error)

    # save extrinsics
    if calib_out_file_name is not None:
        # TODO: Now I think I should have organized this the other way around, but its already baked in so many programs...
        calib = {'K': dict(), 'dist': dict(), 'M': dict()}
        for i, cid in enumerate(cam_ids):
            calib['K']['cam%d' % cid] = K_list[i]
            calib['dist']['cam%d' % cid] = d_list[i]
            calib['M']['cam%d' % cid] = M_list[i]
        calib_file = os.path.join(base_path, calib_out_file_name)
        json_dump(calib_file, calib, verbose=verbose > 0)

    return K_list, d_list, M_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate extrinsic calibration.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data is.')
    parser.add_argument('--cam_pat', type=str, default='cam%d', help='Expression that is being matched '
                                                                     'when searching for camera folders or names.')
    parser.add_argument('--run_pat', type=str, default='run%03d', help='Expression that is being matched '
                                                                       'when searching for runs.')
    parser.add_argument('--estimate_dist', action='store_true', help='Estimate distortion.')
    parser.add_argument('--dist_complexity', type=int, default=2, help='How many distortion parameters to estimate.'
                                                                       ' Should be in [0, 3]')
    parser.add_argument('--det_file_name', type=str, default='detections_cam%d.json',
                        help='File to store detections in.')
    parser.add_argument('--calib_file_name', type=str, default='K_cam%d.json',
                        help='File to load intrinsic calibration from.')
    parser.add_argument('--calib_out_file_name', type=str, default='M.json',
                        help='File to store calibration result in.')
    parser.add_argument('-c', '--cache', action='store_true', help='Use stored version.')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level, higher is more ouput.')
    args = parser.parse_args()

    calc_extrinsics(args.marker, args.data_path,
                    args.cam_pat, args.run_pat, args.det_file_name, args.calib_file_name, args.calib_out_file_name,
                    args.estimate_dist, args.dist_complexity,
                    args.cache, args.verbosity)
