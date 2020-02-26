import os, subprocess
import cv2
import json
import numpy as np

import utils.CamLib as cl
from utils.Graph import *


def _center_extrinsics(cam_extrinsic, object_poses=None, point3d_coord=None):
    """ Enforces the first camera to be the world coordinate system. """
    returnList = list()

    # global transformation
    T = np.linalg.inv(cam_extrinsic[0])

    # calculate new extrinsic matrices
    cam_ext = list()
    for cam_e in cam_extrinsic:
        cam_ext.append(np.matmul(T, cam_e))
    cam_extrinsic = cam_ext
    returnList.append(cam_extrinsic)

    # update object poses
    if object_poses is not None:
        obj_pose_new = list()
        for obj_pose in object_poses:
            if obj_pose is None:
                obj_pose_new.append(None)
            else:
                obj_pose_new.append(np.matmul(T, obj_pose))
        object_poses = obj_pose_new
        returnList.append(object_poses)

    # update 3d points accordingly
    if point3d_coord is not None:
        p3d_h = np.concatenate([point3d_coord,
                                np.ones((point3d_coord.shape[0], 1))],
                               -1)
        point3d_coord = np.matmul(p3d_h, np.transpose(T))
        point3d_coord = point3d_coord[:, :-1] / point3d_coord[:, -1:]
        returnList.append(point3d_coord)

    # in case of a single item get it out of the list
    if len(returnList) == 1:
        returnList = returnList[0]
    return returnList


def calc_3d_object_points(calib_object_points3d, object_poses,
                          point2d_fid, point2d_cid, point2d_mid):
    """ Given the object points in the objects frame and the objects pose this function
    returns the 3D points in the world coordinate frame. """
    assert len(calib_object_points3d.shape) == 2, "Shape mismatch."
    assert calib_object_points3d.shape[0] > 0, "Shape mismatch."
    assert calib_object_points3d.shape[1] == 3, "Shape mismatch."
    assert len(point2d_cid.shape) == 1, "Shape mismatch."
    assert len(point2d_fid.shape) == 1, "Shape mismatch."
    assert len(point2d_mid.shape) == 1, "Shape mismatch."
    assert point2d_cid.shape[0] == point2d_fid.shape[0], "Shape mismatch."
    assert point2d_cid.shape[0] == point2d_mid.shape[0], "Shape mismatch."
    assert point2d_cid.shape[0] >= calib_object_points3d.shape[0], "Shape mismatch."

    num_cams = len(np.unique(point2d_cid))
    num_frames = np.max(point2d_fid) + 1

    point3d_coord = list()
    pid2d_to_pid3d = dict()
    model_points = calib_object_points3d.copy()
    model_points = np.concatenate([model_points,
                                   np.ones((model_points.shape[0], 1))], -1)
    mid_all = [i for i in range(model_points.shape[0])]
    n_start = 0
    for fid in range(num_frames):
        if object_poses[fid] is None:
            continue
        points3d = np.matmul(model_points, np.transpose(object_poses[fid]))
        points3d = points3d[:, :-1] / points3d[:, -1:]
        point3d_coord.append(points3d)

        for cid in range(num_cams):
            mask = np.logical_and(point2d_fid == fid, point2d_cid == cid)  # all 2d points that belong to this frame and camera
            pid2d_ind = np.where(mask)[0]  # p2d id
            mid = point2d_mid[pid2d_ind].tolist() # which markers where detected in the image
            for pid2d, m in zip(pid2d_ind.tolist(), mid):
                p3 = mid_all.index(m)
                pid2d_to_pid3d[pid2d] = n_start + p3  # TODO: there is something going wrong with the indices
        n_start += points3d.shape[0]

    point3d_coord = np.concatenate(point3d_coord, 0)

    return point3d_coord, pid2d_to_pid3d


def estimate_and_score_object_poses(tagpose_estimator, point2d_coord, point2d_cid, point2d_fid, point2d_mid,
                                    cam_intrinsic, cam_dist):

    num_cams = len(cam_intrinsic)
    num_frames = np.max(point2d_fid) + 1
    
    # 0. Score how well a fid/cid pair is suited for estimation of the object pose
    scores_object = dict()
    for fid in range(num_frames):
        scores_object[fid] = dict()
        for cid in range(num_cams):
            scores_object[fid][cid] = 0.0

    # 1. Iterate cams and estimate relative pose to the calibration object for each frame
    T_obj2cam = dict()  # contains the trafo from object points to the cam
    for fid in range(num_frames):
        T_obj2cam[fid] = dict()
        for cid in range(num_cams):
            mask = np.logical_and(point2d_cid == cid, point2d_fid == fid)
            points2d_obs = point2d_coord[mask, :]
            points2d_mid_obs = point2d_mid[mask]

            if np.sum(mask) < 4:
                T_obj2cam[fid][cid] = None
                scores_object[fid][cid] = 0
                continue

            scores_object[fid][cid] = np.sum(mask)  # score is how many points we see there

            points3d_cam, R, t = tagpose_estimator.estimate_relative_cam_pose(cam_intrinsic[cid],
                                                                              cam_dist[cid],
                                                                              points2d_obs, points2d_mid_obs)

            M_object = np.eye(4)
            M_object[:3, :3] = R
            M_object[:3, -1:] = t  # trafo from model to camera frame
            T_obj2cam[fid][cid] = M_object

    return scores_object, T_obj2cam


def greedy_pick_object_pose(scores_object, T_obj2cam, M, verbose):
    object_poses = list()
    for fid in range(len(scores_object)):
        cid = max(scores_object[fid], key=scores_object[fid].get)

        if T_obj2cam[fid][cid] is None:
            if verbose > 1:
                print('- For frame %d estimation of the object pose is impossible' % fid)
            object_poses.append(None)
            continue

        if verbose > 2:
            print('- For frame %d estimating the object pose from camera %d' % (fid, cid))

        # object trafo: x_world = T_object x_model
        T_object = np.matmul(M[cid], T_obj2cam[fid][cid])
        object_poses.append(T_object)

        if verbose > 2:
            print('- Object pose in frame %d:' % fid)
            print('T_object=\n', T_object)
    return object_poses


def _calc_reprojection_error(cam_intrinsic, cam_dist, cam_extrinsic, coord2d_obs, coord3d):
    """ Calculates the reprojection error for a single 2D / 3D point correspondence and its camera calib. """
    if len(coord3d.shape) == 1:
        coord3d = np.expand_dims(coord3d, 0)
    if len(coord2d_obs.shape) == 1:
        coord2d_obs = np.expand_dims(coord2d_obs, 0)

    # transform into this cams frame
    coord3d_h = np.concatenate([coord3d, np.ones((1, 1))], -1)
    coord3d_cam = np.matmul(coord3d_h, np.transpose(np.linalg.inv(cam_extrinsic)))
    coord3d_cam = coord3d_cam[:, :3] / coord3d_cam[:, -1:]

    # calculate projection of 3D point
    coord2d = np.matmul(coord3d_cam, np.transpose(cam_intrinsic))
    coord2d = coord2d[:, :2] / coord2d[:, -1:]

    # apply distortion to the projected point
    coord2d = cl.distort_points(coord2d, cam_intrinsic, cam_dist)

    # find corresponding observation of this cam
    delta_error = np.sqrt(np.sum(np.square(coord2d - coord2d_obs)))
    return delta_error, coord2d


def calculate_reprojection_error(point2d_coord, point3d_coord, pid2d_to_pid3d,
                                 cam_intrinsic, cam_dist, cam_extrinsic,
                                 point2d_cid, show=False, return_cam_wise=False):
    """ Calculate the reprojection error betwee triangulated 3D points and 2D observations. """
    for K in cam_intrinsic:
        assert len(K.shape) == 2, "Shape mismatch."
        assert K.shape[0] == 3, "Shape mismatch."
        assert K.shape[1] == 3, "Shape mismatch."

    for d in cam_dist:
        assert len(d.shape) == 2, "Shape mismatch."
        assert d.shape[0] == 1, "Shape mismatch."
        assert d.shape[1] == 5, "Shape mismatch."

    for M in cam_extrinsic:
        assert len(M.shape) == 2, "Shape mismatch."
        assert M.shape[0] == 4, "Shape mismatch."
        assert M.shape[1] == 4, "Shape mismatch."

    assert len(cam_intrinsic) == len(cam_dist), "Shape mismatch."
    assert len(cam_extrinsic) == len(cam_dist), "Shape mismatch."
    assert len(point2d_coord.shape) == 2, "Shape mismatch."
    assert len(point3d_coord.shape) == 2, "Shape mismatch."
    assert point2d_coord.shape[0] == point2d_cid.shape[0], "Shape mismatch."
    assert point2d_coord.shape[1] == 2, "Shape mismatch."
    assert point3d_coord.shape[1] == 3, "Shape mismatch."

    # calculate the error
    reprojection_error = list()
    reprojection_error_camwise = dict()
    for pid2d, pid3d in pid2d_to_pid3d.items():
        coord3d = np.expand_dims(point3d_coord[pid3d, :], 0)
        coord2d_obs = np.expand_dims(point2d_coord[pid2d, :], 0)
        cid = point2d_cid[pid2d]

        error, _ = _calc_reprojection_error(cam_intrinsic[cid], cam_dist[cid], cam_extrinsic[cid],
                                            coord2d_obs, coord3d)

        reprojection_error.append(error)

        if cid not in reprojection_error_camwise.keys():
            reprojection_error_camwise[cid] = list()
        reprojection_error_camwise[cid].append(error)

    if show:
        print('\n\n------------')
        print('- Reprojection error of %d 3D points onto %d observed points:' % (point3d_coord.shape[0], len(reprojection_error)))
        print('\t> Mean error: %.4f' % np.mean(reprojection_error))
        print('\t> Std. deviation: %.4f' % np.sqrt(np.var(reprojection_error)))
        print('\t> Min/Max: %.4f / %.4f' % (np.min(reprojection_error), np.max(reprojection_error)))

        print('- Mean error per cam:')
        for cid, error_list in reprojection_error_camwise.items():
            print('\t> Cam%d: %.4f (px)' % (cid, np.mean(error_list)))

    if return_cam_wise:
        return np.mean(reprojection_error), reprojection_error_camwise

    return np.mean(reprojection_error)


def estimate_extrinsics_pnp(tagpose_estimator,
                            cam_intrinsic, cam_dist,
                            point2d_coord, point2d_cid, point2d_fid, point2d_pid, point2d_mid,
                            verbose=0):
    """ Estimates extrinsic parameters for each camera from the given 2D point correspondences alone.
        It estimates the essential matrix for camera pairs along the observation graph.

    Input:
        tagpose_estimator: custom object, Estimates the pose between a camera and the calibration objects.
        cam_intrinsic: list of 3x3 np.array, Intrinsic calibration of each camera.
        cam_dist: list of 1x5 np.array, Distortion coefficients following the OpenCV pinhole camera model.
        point2d_coord: Nx2 np.array, Array containing 2D coordinates of N points.
        point2d_cid: Nx1 np.array, Array containing the camera id for each of the N points.
        point2d_fid: Nx1 np.array, Array containing the frame id for each of the N points.
        point2d_pid: Nx1 np.array, Array containing a unique point id for each of the N points.
        point2d_mid: Nx1 np.array, Array containing a marker-unique id for each of the N points.

    Returns:
        cam_extrinsic: list of 4x4 np.array, Intrinsic calibration of each camera.
        calib_object_points3d: Mx3 np.array, 3D Points of the calibration object in a object based frame.
    """
    assert len(cam_intrinsic) >= 2, "Too little cameras."
    assert len(cam_intrinsic) == len(cam_dist), "Shape mismatch."
    assert len(point2d_cid.shape) == 1, "Shape mismatch."
    assert len(point2d_fid.shape) == 1, "Shape mismatch."
    assert len(point2d_pid.shape) == 1, "Shape mismatch."
    assert len(point2d_mid.shape) == 1, "Shape mismatch."
    assert point2d_coord.shape[0] == point2d_cid.shape[0], "Shape mismatch."
    assert point2d_coord.shape[0] == point2d_fid.shape[0], "Shape mismatch."
    assert point2d_coord.shape[0] == point2d_pid.shape[0], "Shape mismatch."
    assert point2d_coord.shape[0] == point2d_mid.shape[0], "Shape mismatch."
    assert len(cam_intrinsic) == len(np.unique(point2d_cid).flatten().tolist()), "Shape mismatch."

    if verbose > 0:
        print('\n\n------------')
        print('- Estimating extrinsic parameters by solving PNP problems')

    num_cams = len(cam_intrinsic)
    num_frames = np.max(point2d_fid) + 1

    # get model shape
    calib_object_points3d = tagpose_estimator.object_points.copy()

    # 1. Iterate cams and estimate relative pose to the calibration object for each frame
    scores_object, T_obj2cam = estimate_and_score_object_poses(tagpose_estimator,
                                                               point2d_coord, point2d_cid,
                                                               point2d_fid, point2d_mid,
                                                               cam_intrinsic, cam_dist)

    def _calc_score(T1, T2):
        """ Estimates how closely T1 * T2 = eye() holds. """
        R = np.matmul(T1, T2) - np.eye(T1.shape[0])
        return np.sum(np.abs(R))  # frobenius norm

    # try to find the pair of frames which worked best --> estimate relative camera pose from there
    scores_rel_calib = dict()  # store how good this guess seems to be for calibrating a cam pair
    for fid1 in range(num_frames): # for each pair of frames
        for fid2 in range(fid1, num_frames):
            scores_rel_calib[fid1, fid2] = dict()
            for cid1 in range(num_cams):  # check each pair of cams
                for cid2 in range(cid1+1, num_cams):
                    # check if its valid
                    if (T_obj2cam[fid1][cid2] is None) or (T_obj2cam[fid1][cid1] is None) or \
                            (T_obj2cam[fid2][cid2] is None) or (T_obj2cam[fid2][cid1] is None):
                        s_rel = float('inf')
                    else:

                        # calculate the transformation cam1 -> cams2 using fid1
                        T12_fid1 = np.matmul(T_obj2cam[fid1][cid2],
                                             np.linalg.inv(T_obj2cam[fid1][cid1]))

                        # calculate the transformation cam2 -> cams1 using fid2
                        T21_fid2 = np.matmul(T_obj2cam[fid2][cid1],
                                             np.linalg.inv(T_obj2cam[fid2][cid2]))

                        # for perfect estimations the two mappings should be the inverse of each others
                        s_rel = _calc_score(T12_fid1, T21_fid2)

                    scores_rel_calib[fid1, fid2][cid1, cid2] = s_rel

    # 3. Find out which frames are optimal for a given cam pair
    cam_pair_best_fid = dict()
    for cid1 in range(num_cams):
        for cid2 in range(cid1+1, num_cams):
            min_fid = None
            min_v = float('inf')
            for fid_pair, score_dict in scores_rel_calib.items():
                # get an initial value
                if min_fid is None:
                    min_fid = fid_pair
                    min_v = score_dict[cid1, cid2]
                    continue

                # if current best is worse than current item replace
                if min_v > score_dict[cid1, cid2]:
                    min_fid = fid_pair
                    min_v = score_dict[cid1, cid2]

            cam_pair_best_fid[cid1, cid2] = min_fid

    # 3. Build observation graph and use djikstra to estimate relative camera poses
    observation_graph = Graph()
    for cid in range(num_cams):
        observation_graph.add_node(cid)

    # populate with edges
    score_accumulated = [0 for _ in range(num_cams)]  # accumulate score for each cam
    for cid1 in range(num_cams):
        for cid2 in range(cid1+1, num_cams):
            fid_pair = cam_pair_best_fid[cid1, cid2]
            s = scores_rel_calib[fid_pair][cid1, cid2]
            observation_graph.add_edge(cid1, cid2, s)
            observation_graph.add_edge(cid2, cid1, s)
            score_accumulated[cid1] += s
            score_accumulated[cid2] += s

    # root cam (the one that has the lowest overall score)
    root_cam_id = np.argmin(np.array(score_accumulated))

    if verbose > 1:
        print('- Accumulated score (lower is better): ', score_accumulated)
        print('- Choosing root cam', root_cam_id)

    # 4. Determine which relative poses to estimate
    # use Dijkstra to find "cheapest" path (i.e. the one with most observations) from the starting cam to all others
    cam_path = dict()  # contains how to get from the starting cam to another one cam_path[target_cam] = [path]
    for cid in range(num_cams):
        if cid == root_cam_id:
            cam_path[cid] = [cid]
            continue
        cost, camchain = shortest_path(observation_graph, root_cam_id, cid)
        cam_path[cid] = camchain

    if verbose > 1:
        for k, v in cam_path.items():
            print('- Camchain to %d: ' % k, v)

    # 5. Put together the relative camera poses
    relative_pose = dict()  # contains the trafo from start -> target
    relative_pose_pair = dict()  # contains already estimated poses between cameras;
    # is the trafo from j -> i;  xi = relative_pose_pair[i, j] * xj
    for target_camid in range(num_cams):
        if target_camid == root_cam_id:
            relative_pose[target_camid] = np.eye(4)
            continue

        M = np.eye(4)  # this is the trafo from start -> X
        for i in range(len(cam_path[target_camid] ) -1):  # traverse cam path
            # get current cam_pair on the cam_path
            inter_camid1 = cam_path[target_camid][i]  # this is where we currently are
            inter_camid2 = cam_path[target_camid][ i +1]  # this is where we want to transform to

            swapped = False
            if inter_camid2 < inter_camid1:
                t = inter_camid2
                inter_camid2 = inter_camid1
                inter_camid1 = t
                swapped = True

            if verbose > 1:
                print('- Attempting to estimate the relative pose from cam %d --> %d' % (inter_camid1, inter_camid2))

            # calculate only when not calculated yet
            if (inter_camid1, inter_camid2) not in relative_pose_pair.keys():
                fid1, fid2 = cam_pair_best_fid[inter_camid1, inter_camid2]

                msg = "Calibration impossible! There is no way feasible way to calibrate cam%d and cam%d." % \
                (inter_camid1, inter_camid2)
                assert T_obj2cam[fid1][inter_camid1] is not None, msg
                assert T_obj2cam[fid1][inter_camid2] is not None, msg

                # calculate the transformation cam1 -> cams2 using the optimal fids
                T12 = np.matmul(T_obj2cam[fid1][inter_camid1],
                                np.linalg.inv(T_obj2cam[fid1][inter_camid2]))
                relative_pose_pair[inter_camid1, inter_camid2] = T12

            delta = relative_pose_pair[inter_camid1, inter_camid2]
            if swapped:
                delta = np.linalg.inv(delta)

            # accumulate trafos
            M = np.matmul(delta, M)
        relative_pose[target_camid] = M

    if verbose > 0:
        print('- Extrinsics estimated')

    if verbose > 2:
        for cid in range(num_cams):
            print('\n- Trafo Root (%d) --> %d' % (root_cam_id, cid))
            print(relative_pose[cid])
            print('')

    cam_extrinsic = list()
    for cid in range(num_cams):
        cam_extrinsic.append(relative_pose[cid])

    # 6. Figure out the object poses (if there is no observation its impossible)
    object_poses = greedy_pick_object_pose(scores_object, T_obj2cam, relative_pose, verbose)

    cam_extrinsic, object_poses = _center_extrinsics(cam_extrinsic,
                                                     object_poses)  # ensure camera 0 is the world center
    point3d_coord, pid2d_to_pid3d = calc_3d_object_points(calib_object_points3d, object_poses,
                                                          point2d_fid, point2d_cid, point2d_mid)
    return cam_extrinsic, point3d_coord, pid2d_to_pid3d, object_poses


def _dump_bal_json_pnp(file_path,
                       cam_intrinsic, cam_dist, cam_extrinsic,
                       calib_object_points3d, object_poses, img_shapes,
                       point2d_coord, point2d_cid, point2d_fid, point2d_mid,
                       verbose):
    """ Writes data to file_path as json file, which can be read by my ceres BundleAdjuster. """
    num_cams = len(cam_intrinsic)
    num_obs = point2d_coord.shape[0]

    # write camera parameters
    cameraList = list()
    for cid in range(num_cams):
        cam = list()
        K = cam_intrinsic[cid]
        cam.append(float(K[0, 0]))  #fx
        cam.append(float(K[1, 1]))  #fy
        cam.append(float(K[0, 2]))  #ppx
        cam.append(float(K[1, 2]))  #ppy
        cam.append(float(cam_dist[cid][0, 0]))  #rad1
        cam.append(float(cam_dist[cid][0, 1]))  #rad2
        cam.append(float(cam_dist[cid][0, 2]))  #tang1
        cam.append(float(cam_dist[cid][0, 3]))  #tang2
        cam.append(float(cam_dist[cid][0, 4]))  #rad3

        M = cam_extrinsic[cid]
        M = np.linalg.inv(M)
        t_r = M[:3, -1]
        t = t_r
        r, _ = cv2.Rodrigues(M[:3, :3])
        cam.append(float(r[0]))  #r1
        cam.append(float(r[1]))  #r2
        cam.append(float(r[2]))  #r3
        cam.append(float(t[0]))  #tx
        cam.append(float(t[1]))  #ty
        cam.append(float(t[2]))  #tz

        cam.append(int(img_shapes[cid][1]))  #width
        cam.append(int(img_shapes[cid][0]))  #height

        cameraList.append(cam)

    # write model points
    modelPointList = list()
    for point3d in calib_object_points3d:
        modelPointList.append([float(point3d[0]), float(point3d[1]), float(point3d[2])])

    # write object poses
    obj_poses = list()
    for T in object_poses:
        if T is None:
            T = np.eye(4)
        obj = list()
        R = T[:3, :3]
        t = T[:, -1:]
        r, _ = cv2.Rodrigues(R)
        obj.append(float(r[0]))  #r1
        obj.append(float(r[1]))  #r2
        obj.append(float(r[2]))  #r3
        obj.append(float(t[0]))  #tx
        obj.append(float(t[1]))  #ty
        obj.append(float(t[2]))  #tz
        obj_poses.append(obj)

    # write 2d observations
    observedPointList, observedPointPointId, observedPointCamId, observedPointFrameId = list(), list(), list(), list()
    for pid2d in range(num_obs):
        # check if object pose is available
        if object_poses[int(point2d_fid[pid2d])] is None:
            fid = point2d_fid[pid2d]
            cid = point2d_cid[pid2d]
            mask = np.logical_and(point2d_cid == cid, point2d_fid == fid)
            assert object_poses[fid] is not None, "should not happen"

        observedPointList.append([float(point2d_coord[pid2d, 0]),
                                  float(point2d_coord[pid2d, 1])])
        observedPointPointId.append(int(point2d_mid[pid2d]))
        observedPointCamId.append(int(point2d_cid[pid2d]))
        observedPointFrameId.append(int(point2d_fid[pid2d]))

    data_dict = {'Camera': cameraList,
                 'ModelPoints': modelPointList,
                 'ObjectPoses': obj_poses,
                 'ObservedPoints': { 'coords': observedPointList,
                                     'pid': observedPointPointId,
                                     'cid': observedPointCamId,
                                     'fid': observedPointFrameId} }

    with open(file_path, 'w') as fo:
        json.dump(data_dict, fo, sort_keys=True, indent=4)

    if verbose:
        print('Saved problem as: %s' % file_path)


def load_json_pnp(file_path, verbose):
    """ Loads 3d points and cameras from a json file. """
    with open(file_path, 'r') as fi:
        data_dict = json.load(fi)

    if verbose:
        print('- Loaded data from: %s' % file_path)

    cam_intrinsic = list()
    cam_extrinsic = list()
    cam_dist = list()

    # cameras
    cameraList = data_dict['Camera']
    for cid, cam_data in enumerate(cameraList):
        K = np.eye(3)
        K[0, 0] = cam_data[0]
        K[1, 1] = cam_data[1]
        K[0, 2] = cam_data[2]
        K[1, 2] = cam_data[3]
        cam_intrinsic.append(K)

        dist = np.array([cam_data[4], cam_data[5], cam_data[6], cam_data[7], cam_data[8]])
        cam_dist.append(np.expand_dims(dist, 0))

        R, _ = cv2.Rodrigues(np.array([cam_data[9], cam_data[10], cam_data[11]]))
        M = np.eye(4)
        M[:3, :3] = R
        t = np.array([cam_data[12], cam_data[13], cam_data[14]])
        M[:3, 3] = t
        cam_extrinsic.append(np.linalg.inv(M))

    # world points
    object_poses = list()
    objectPoses = data_dict['ObjectPoses']
    for pose in objectPoses:
        r = np.array(pose[:3])
        t = np.array(pose[3:])
        R, _ = cv2.Rodrigues(r)
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = t
        object_poses.append(M)

    return cam_intrinsic, cam_dist, cam_extrinsic, object_poses


def run_bundle_adjust_pnp(cam_intrinsic, cam_dist, cam_extrinsic,
                          point2d_coord, point2d_cid, point2d_fid, point2d_mid,
                          calib_object_points3d, object_poses, img_shapes,
                          optimize_intrinsic=True, optimize_distortion=True,
                          optimize_extrinsic=True, shared_camera_model=False,
                          verbose=0):
    """ Run bundle adjustment. """

    if verbose > 0:
        print('\n\n------------')
        print('- Running bundle adjustment on PNP problem to optimize parameters jointly')

    out_file = './guess.json'
    in_file = './optim.json'

    _dump_bal_json_pnp(out_file,
                       cam_intrinsic, cam_dist, cam_extrinsic,
                       calib_object_points3d, object_poses, img_shapes,
                       point2d_coord, point2d_cid, point2d_fid, point2d_mid,
                       verbose)

    command = list()
    path_to_this_file = os.path.dirname(os.path.realpath(__file__))
    command.append(os.path.join(path_to_this_file, '../Bundle/build/ceres_librarypnp'))
    if optimize_intrinsic:
        command.append('-k')
    if optimize_distortion:
        command.append('-r')
    if optimize_extrinsic:
        command.append('-m')
    if shared_camera_model:
        command.append('-s')
    command.append('-i%s' % out_file)
    command.append('-o%s' % in_file)

    # Call bundle adjust program
    if verbose == 0:
        subprocess.call(command, stdout=open(os.devnull, 'wb'))
    else:
        subprocess.call(command)

    cam_intrinsic, cam_dist, cam_extrinsic, object_poses_new = load_json_pnp(in_file, verbose)

    # replace invalid object poses with None
    object_poses_new2 = list()
    for old, new in zip(object_poses, object_poses_new):
        if old is None:
            object_poses_new2.append(None)
        else:
            object_poses_new2.append(new)
    object_poses = object_poses_new2

    cam_extrinsic, object_poses = _center_extrinsics(cam_extrinsic=cam_extrinsic, object_poses=object_poses)
    point3d_coord, _ = calc_3d_object_points(calib_object_points3d, object_poses,
                                             point2d_fid, point2d_cid, point2d_mid)
    os.remove(out_file)
    os.remove(in_file)

    return cam_intrinsic, cam_dist, cam_extrinsic, point3d_coord
