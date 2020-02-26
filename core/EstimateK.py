import numpy as np
import time
import cv2
from scipy.spatial import ConvexHull
from PIL import Image, ImageDraw


def estimate_intrinsics(point2d_coord, point2d_fid, model_point3d_coord, img_shape,
                        estimate_dist=True, dist_complexity=2, verbose=1):
    """ Estimates intrinsic parameters for each camera from the given 2D point correspondences.

    Input:
        point2d_coord: Nx2 np.array, Array containing 2D coordinates of N points.
        point2d_fid: Nx2 np.array, Array containing the frame id for each of the N points.
        model_point3d_coord: Nx3 np.array, Array containing the 3D coordinates in a marker based coordinate system for each of the N points.
        img_shape: uple of two int, Shapes of the images (height1, width1).
        estimate_dist: bool, If the distortion parameters should be estimated or not.
        dist_complexity: int, Level of complexity of the distortion model; 1 = K1, 2 = K1 + K2, else = K1 + K2 + K3

    Returns:
        cam_intrinsic: list of 3x3 np.array, Intrinsic calibration of each camera.
        cam_dist: list of 1x5 np.array, Distortion coefficients following the OpenCV pinhole camera model.
    """
    assert point2d_coord.shape[0] == point2d_fid.shape[0], "Shape mismatch."
    assert point2d_coord.shape[0] == model_point3d_coord.shape[0], "Shape mismatch."
    assert point2d_coord.shape[1] == 2, "Shape mismatch."
    assert model_point3d_coord.shape[1] == 3, "Shape mismatch."

    if verbose > 0:
        print('------------')
        print('- Estimating intrinsic parameters')

    if verbose > 0:
        startTime = time.time()

    # accumulate points
    img_points, object_points = list(), list()
    pts_count, normal_list, hull_points_list = list(), list(), list()
    for fid in np.unique(point2d_fid).tolist():
        mask = np.array(point2d_fid) == fid
        if np.sum(mask) < 4:
            continue

        # subset of visible 2d/3d points
        points2d_obs = point2d_coord[mask, :]
        model_point3d_coord_obs = model_point3d_coord[mask, :]

        img_points.append(points2d_obs)
        object_points.append(model_point3d_coord_obs)

        # Coarse K approximation: It uses the relation between FOV alpha, sensor size w and focal length f:
        # f = 1/ (2* tan[alpha/2]) * w
        # assuming a FOV of 60deg -> f = 0.86 * w
        # Another assumption is that the principal point is in the middle of the image
        K_guess = np.array([[img_shape[1] * 0.86, 0.0, img_shape[1] * 0.5],
                            [0.0, img_shape[0] * 0.86, img_shape[0] * 0.5],
                            [0.0, 0.0, 1.0]])
        success, r_rel, _ = cv2.solvePnP(np.expand_dims(model_point3d_coord_obs, 1),
                                         np.expand_dims(points2d_obs, 1),
                                         K_guess, distCoeffs=np.zeros((5,)), flags=cv2.SOLVEPNP_ITERATIVE)

        # get normal vector from tag rotation
        R, _ = cv2.Rodrigues(r_rel)
        n = np.matmul(R, np.array([0.0, 0.0, 1.0]))
        normal_list.append(n)

        # figure out non axis aligned bounding box
        hull = ConvexHull(points2d_obs)
        hull_points = points2d_obs[hull.vertices]
        hull_points /= np.array([[img_shape[1], img_shape[0]]], dtype=np.float32)
        hull_points_list.append(hull_points)
        pts_count.append(np.sum(mask))

    if verbose > 0:
        print('- For estimating intrinsic there are %d'
              ' 2D->3D correspondences' % (sum([x.shape[0] for x in img_points])))

    # parameters of the selection algorithm
    img_size = 100
    normal_thresh = 10.0  # when normals differ more than that angle we add it (angle in deg)
    area_thresh = 0.1  # when new area is more than that we add it (percentage of image area)

    def _score_normals(n1, n2):
        """ Calculates the angle between two normals. """
        return np.arccos(np.dot(n1, n2))

    def _score_area(area_covered, hullpts, tmp_img_size=img_size):
        """ Calculates the percentage of new area hullpts are adding to the current area covered. """
        # scale points to tmp size
        hullpts = hullpts.copy() * tmp_img_size

        # draw tmp images
        hullpts1 = hullpts.astype(np.int32).reshape([-1]).tolist()
        map = Image.new('L', (tmp_img_size, tmp_img_size), 0)
        ImageDraw.Draw(map).polygon(hullpts1, outline=1, fill=1)

        # calculate how much is new
        percentage_new = 1.0 - np.sum(np.logical_and(map, area_covered)) / (np.sum(map) + 1e-6)
        return percentage_new

    def _hull2mask(hullpts, tmp_img_size=100):
        """ Converts a convex hull (set of points) into an binary mask. """
        # scale points to tmp size
        hullpts = hullpts.copy() * tmp_img_size

        # draw tmp images
        hullpts1 = hullpts.astype(np.int32).reshape([-1]).tolist()
        mask = Image.new('L', (tmp_img_size, tmp_img_size), 0)
        ImageDraw.Draw(mask).polygon(hullpts1, outline=1, fill=1)
        return mask

    # select a good subset from the images we have
    ind_selected = list()  # subset of views we use

    # sort by number of points visibile in view
    sort_ind = np.argsort(pts_count)[::-1]

    # greedily pick views
    area_covered = np.zeros((img_size, img_size))
    for i in sort_ind.tolist():
        if pts_count[i] < 8:
            # when there are too little point on the plane the normal estimation usually is bad
            continue

        # check how close this normal is to any we already selected
        if len(ind_selected) > 0:
            score_n = min([_score_normals(normal_list[j], normal_list[i]) for j in ind_selected]) * 180.0 / np.pi
        else:
            score_n = 0.0
        # print('Difference to closest angle in selected set', score_n)

        if score_n > normal_thresh:
            # print('Added %d due to normal condition' % i)
            ind_selected.append(i)
            area_covered = np.logical_or(area_covered, _hull2mask(hull_points_list[i]))
        else:
            score_a = _score_area(area_covered, hull_points_list[i])
            # print('percentage_new', score_a)
            if score_a > area_thresh:
                # print('Added %d due to area condition' % i)
                ind_selected.append(i)
                area_covered = np.logical_or(area_covered, _hull2mask(hull_points_list[i]))

        # find out largest angle difference
        angle_max = 0.0
        for k1, ind1 in enumerate(ind_selected):
            for k2 in range(k1 + 1, len(ind_selected)):
                ind2 = ind_selected[k2]
                angle_max = max(angle_max, _score_normals(normal_list[ind1], normal_list[ind2]))
        angle_max *= 180.0 / np.pi

        # stop when image is mostly covered and there is some minimal angular difference
        if (np.mean(area_covered) > 0.8) and (angle_max > 30.0):
            break

    if verbose > 0:
        print('- Estimating intrinsic for from subset of %d views yielding %d'
              ' 2D->3D correspondences' % (len(ind_selected), sum([pts_count[i] for i in ind_selected])))

    # take subset
    object_points = [object_points[i].astype(np.float32) for i in ind_selected]
    img_points = [img_points[i].astype(np.float32) for i in ind_selected]

    if verbose > 0:
        print('- Cam: For estimating intrinsic there are %d'
              ' 2D->3D correspondences' % (sum([x.shape[0] for x in img_points])))

    # find initial solution
    K_init = cv2.initCameraMatrix2D(object_points,
                                    img_points,
                                    (img_shape[1], img_shape[0]))

    if verbose > 2:
        print('- Cam K_init:')
        print(K_init)

    # estimate intrinsics for this cam
    # scaling error up with number of points; otherwise this function takes forever when there are many points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100,
                0.001)  # accuracy should scale with number of points used
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    # disable most of the distortion parameters by default, because estimation is usually bad
    if dist_complexity == 1:
        flags += cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + \
                 cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    elif dist_complexity == 2:
        flags += cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3 + \
                 cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    else:
        flags += cv2.CALIB_ZERO_TANGENT_DIST + \
                 cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6

    if not estimate_dist:
        # also turn off all factors
        flags = cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_ZERO_TANGENT_DIST + \
                cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3 + \
                cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6

    error, K, dist, _, _ = cv2.calibrateCamera(object_points, img_points,
                                               (img_shape[1], img_shape[0]), K_init, None,
                                               criteria=criteria,
                                               flags=flags)

    if verbose > 2:
        print('K')
        print(K, '\n')
        print('dist')
        print(dist, '\n')

    if verbose > 2:
        print('- Estimating intrinsics took %.2f seconds' % (time.time() - startTime))

    if verbose > 0:
        print('- Cam: Reprojection error over %d points: %.3f' % (
        sum([x.shape[0] for x in img_points]), error))
        print('------------')

    return K, dist
