import argparse, os
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm

from core.BoardDetector import BoardDetector
from utils.general_util import find_images, json_load, fig2data
import utils.CamLib as cl


def _calc_board_stats(points2d_obs, model_point3d_coord_obs, K, dist, img=None):
    success, r_rel, t_rel = cv2.solvePnP(np.expand_dims(model_point3d_coord_obs, 1),
                                         np.expand_dims(points2d_obs, 1),
                                         K, distCoeffs=dist, flags=cv2.SOLVEPNP_ITERATIVE)

    # get normal vector from tag rotation
    R, _ = cv2.Rodrigues(r_rel)
    n = np.matmul(R, np.array([0.0, 0.0, 1.0]))  # normal wrt camera
    n = np.clip(np.sum(n), -1.0, 1.0)
    angle = np.arccos(n)*180.0/np.pi

    # calculate points in camera frame
    M_w2c = np.concatenate([R, t_rel], 1)
    M_w2c = np.concatenate([M_w2c, np.array([[0.0, 0.0, 0.0, 1.0]])], 0)
    p3d = cl.trafo_coords(model_point3d_coord_obs, M_w2c)

    # reprojection error
    p2d_p = cl.project(p3d, K, dist)
    err = np.linalg.norm(p2d_p - points2d_obs, 2, -1)

    if img is not None:
        import matplotlib.pyplot as plt
        plt.imshow(img[:, :, ::-1])
        plt.plot(points2d_obs[:, 0], points2d_obs[:, 1], 'go')
        plt.plot(p2d_p[:, 0], p2d_p[:, 1], 'rx')
        plt.show()
    return angle, p3d[:, -1], err


def _calc_hist(points, max_x, max_y, num_bins):
    hist = np.zeros((num_bins, num_bins))

    step = points / np.array([[max_y, max_x]]) * (num_bins-1)
    bins = np.floor(step).astype(np.int32)

    for x, y in bins:
        hist[y, x] += 1

    hist /= hist.max()

    return hist


def _show_coverage(img, pts_all, show_size):
    # calculate 2d histogram
    hist2d = _calc_hist(pts_all, *img.shape[:2], 20)
    hist2d = cv2.resize(hist2d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    hist2d = cm.jet(hist2d)[:, :, :3]*255
    merged = (0.5*img.astype(np.float32) + 0.5*hist2d.astype(np.float32)).astype(np.uint8)

    # draw all of them as dots
    for p in pts_all:
        p = p.round().astype(np.int32)
        c = (p[0], p[1])
        cv2.circle(img, center=c, radius=2, color=(0, 0, 255))
    merged = np.concatenate([img, merged], 1)
    s = int(round(img.shape[0]/float(img.shape[1])*show_size))
    merged = cv2.resize(merged, (show_size, s))
    cv2.imshow('coverage', merged)
    cv2.waitKey()


def _read_first_frame(data_path, det):
    if os.path.isdir(data_path):
        # check for image files
        img_list = find_images(data_path)
        print('Found %s images.' % len(img_list))

        # sanity check
        assert len(det['p2d']) == len(img_list), 'Number of detections and number of images differs.'
        assert len(det['pid']) == len(img_list), 'Number of detections and number of images differs.'
        assert len(det['files']) == len(img_list), 'Number of detections and number of images differs.'

        img = cv2.imread(img_list[0])

    else:
        cap = cv2.VideoCapture(data_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Found video with %s frames.' % num_frames)

        # sanity check
        assert len(det['p2d']) == num_frames, 'Number of detections and number of images differs.'
        assert len(det['pid']) == num_frames, 'Number of detections and number of images differs.'
        assert len(det['pid']) == num_frames, 'Number of detections and number of images differs.'

        ret = False
        while not ret:
            ret, img = cap.read()

    return img


def show_intrinsic_calib(marker_path, data_path, det_file_name, calib_file_name, show_size):
    print('Showing marker detections for:')
    print('\tData path: %s' % data_path)
    print('\tMarker file: %s' % marker_path)

    if os.path.isdir(data_path):
        print('\tAssuming: Folder of images.')
        base_dir = data_path
    else:
        print('\tAssuming: Video file.')
        base_dir = os.path.dirname(data_path)

    det_file = os.path.join(base_dir, det_file_name)
    calib_file = os.path.join(base_dir, calib_file_name)
    print('\tDetection file: %s' % det_file)
    print('\tCalib file: %s' % calib_file)

    # load detections
    assert os.path.exists(det_file), 'Could not find detection file.'
    det = json_load(det_file)

    # load calibration
    assert os.path.exists(calib_file), 'Could not find detection file.'
    calib = json_load(calib_file)
    K, dist = np.array(calib['K']), np.array(calib['dist'])

    img = _read_first_frame(data_path, det)

    # set up detector
    detector = BoardDetector(marker_path)

    # calculate statistics
    err, angle, depths = list(), list(), list()
    for p2d, pid in tqdm(zip(det['p2d'], det['pid']),
                         total=len(det['p2d']), desc='Calculating stats'):
        if len(pid) == 0:
            continue

        p3d_m = detector.object_points[pid]
        a, d, e = _calc_board_stats(np.array(p2d),
                                    p3d_m,
                                    K, dist)
        angle.append(a)
        depths.extend(d)
        err.extend(e)

    # Print reprojection error
    err = np.array(err)
    print('Reprojection error: min=%.2f, mean=%.2f, max=%.2f (px)' % (err.min(), err.mean(), err.max()))

    # show error and depth distribution
    angle = np.array(angle)
    angle = angle[~np.isnan(angle)]
    depths = np.array(depths)

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.hist(depths), ax1.set_title('distance to camera'), ax1.set_xlim([0, depths.max()])
    ax2.hist(angle), ax2.set_title('angle wrt camera'), ax2.set_xlim([0, 180])
    stats_img = fig2data(fig)
    cv2.imshow('stats', stats_img[:, :, ::-1])
    cv2.waitKey(100)

    # show image space coverage
    pts_all = np.concatenate([d for d in det['p2d'] if len(d) > 0], 0)
    print('Showing %d detected points' % pts_all.shape[0])
    _show_coverage(img, pts_all, show_size)

    return det


if __name__ == "__main__":
    """
        Visualize intrinsic calibration process:
            - Show which areas are being covered by correspondences
            - How the normals and distances are distributed
            - Calculate a reprojection error
    """
    parser = argparse.ArgumentParser(description='Visualize intrinsic calibration sequence.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data and the detection file is.')
    parser.add_argument('--show_size', type=int, default=640, help='Width of image shown')
    parser.add_argument('--det_file_name', type=str, default='detections.json',
                        help='File detections are stored in.')
    parser.add_argument('--calib_file_name', type=str, default='K.json',
                        help='File intrinsic calibration is stored in.')
    args = parser.parse_args()

    show_intrinsic_calib(args.marker, args.data_path, args.det_file_name, args.calib_file_name, args.show_size)
