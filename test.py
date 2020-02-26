import os
import numpy as np

from utils.general_util import json_load

def _same(a, b, rtol=1.e-5, atol=1.e-8):
    a = np.array(a)
    b = np.array(b)

    assert a.shape == b.shape, 'Shape mismatch.'
    assert np.allclose(a, b, rtol=rtol, atol=atol), 'Value mismatch.'


def _draw_tag(ax, point_list, linewidth=2, cross_out=False, tagId=None):
    # print(point_list[0])
    lines = [(0, 1, 'c'), (1, 2, 'g'), (2, 3, 'g'), (3, 0, 'g')]  # bottom, right, top, left (goes around CCW)
    for l in lines:
        ax.plot([point_list[l[0]][0], point_list[l[1]][0]],
                [point_list[l[0]][1], point_list[l[1]][1]], l[2], linewidth=linewidth)

    if cross_out:
        ax.plot([point_list[0][0], point_list[2][0]],
                [point_list[0][1], point_list[2][1]], 'r', linewidth=linewidth)
        ax.plot([point_list[1][0], point_list[3][0]],
                [point_list[1][1], point_list[3][1]], 'r', linewidth=linewidth)

    if tagId is not None:
        middle = np.mean(np.array(point_list), 0)
        ax.text(middle[0], middle[1], '%d' % tagId, fontsize=25, color='r', horizontalalignment='center', verticalalignment='center')


def test_tag_detector(show=False):
    """ Test detecting Apriltags in images. """
    from TagDetector.AprilTagDetectorBatch import PyRunAprilDetectorBatch

    detector = PyRunAprilDetectorBatch('36h11', 2, 1, 1.0, False)
    img_list = ['./data/calib_test_data/real/aprilboard_sample.png']
    assert os.path.exists(img_list[0]), 'Image file not found.'
    det_list = detector.processImageBatch(img_list)

    gt = {6: [(145.8004608154297, 292.304443359375), (227.15277099609375, 293.6097106933594), (226.2543182373047, 224.91122436523438), (146.40545654296875, 215.8142547607422)], 7: [(242.0404052734375, 293.93133544921875), (305.8508605957031, 294.74029541015625), (304.23236083984375, 233.81527709960938), (241.2886962890625, 226.5770721435547)], 8: [(318.0124816894531, 295.3065185546875), (368.8752136230469, 295.9106140136719), (367.4349670410156, 240.8336944580078), (316.3474426269531, 235.1453399658203)], 4: [(243.25547790527344, 377.8477478027344), (307.7796325683594, 371.12799072265625), (306.4921569824219, 307.4539489746094), (242.35556030273438, 307.6281433105469)], 3: [(145.40650939941406, 387.8465576171875), (227.60450744628906, 379.4621887207031), (227.33518981933594, 307.6053771972656), (145.74053955078125, 308.3574523925781)], 5: [(320.2066650390625, 369.9797058105469), (371.4264221191406, 364.89306640625), (369.5838928222656, 307.49066162109375), (318.6220397949219, 307.4967041015625)], 2: [(322.03857421875, 447.73651123046875), (374.1725769042969, 435.9242858886719), (371.84027099609375, 376.7850646972656), (320.30230712890625, 383.0290222167969)], 1: [(243.34136962890625, 465.0435791015625), (309.4181823730469, 450.6374816894531), (308.1007385253906, 384.43450927734375), (243.43826293945312, 392.24005126953125)], 0: [(144.28009033203125, 487.2640380859375), (227.9747314453125, 468.9134826660156), (227.88999938964844, 394.06170654296875), (144.9755859375, 404.3989562988281)]}

    # check result
    assert len(det_list[0]) == len(gt), 'Number of detected tags differs.'
    for det in det_list[0]:
        assert np.allclose(
            np.array(det.points),
            np.array(gt[det.id])
        ), 'Results changed.'

    if show:
        import matplotlib.pyplot as plt
        import scipy.misc

        for img_path, det_img in zip(img_list, det_list):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            img = scipy.misc.imread(img_path)
            ax.imshow(img, cmap='gray')

            for det in det_img:
                _draw_tag(ax, det.points, tagId=det.id)
            plt.show()

    print('SUCCESS: test_tag_detector')


def test_board_pose_estimator(show=False):
    """ Test detecting Boards in images. """
    import cv2
    from utils.general_util import json_load
    from core.BoardDetector import BoardDetector
    img_list = ['./data/calib_test_data/real/april_board_tags_sample.JPG']
    assert os.path.exists(img_list[0]), 'Image file not found.'

    detector = BoardDetector('./data/calib_test_data/marker_32h11b2_4x4x_7cm.json')
    point_coords_frames, point_ids_frames = detector.process_image_batch(img_list)
    gt1 = json_load('data/calib_test_data/real/gt_det1.json')
    _same(point_coords_frames[0], gt1['c'])
    _same(point_ids_frames[0], gt1['i'])

    if show:
        for img_path, points, point_ids in zip(img_list, point_coords_frames, point_ids_frames):
            image = cv2.imread(img_path)
            detector.draw_board(image, points, point_ids, linewidth=2)

    detector = BoardDetector('./data/calib_test_data/marker_16h5b1_4x4x_15cm.json')
    point_coords_frames, point_ids_frames = detector.process_image_batch(img_list)
    gt2 = json_load('data/calib_test_data/real/gt_det2.json')
    _same(point_coords_frames[0], gt2['c'])
    _same(point_ids_frames[0], gt2['i'])

    if show:
        for img_path, points, point_ids in zip(img_list, point_coords_frames, point_ids_frames):
            image = cv2.imread(img_path)
            detector.draw_board(image, points, point_ids, linewidth=2)

    print('SUCCESS: test_board_pose_estimator')


def test_calib_K_no_dist():
    from calib_K import calc_intrinsics
    K, dist = calc_intrinsics('./data/calib_test_data/marker_32h11b2_4x4x_7cm.json',
                              'data/calib_test_data/rendered/K_test/cam0/',
                              det_file_name=None,
                              estimate_dist=False)

    calib_gt = json_load('data/calib_test_data/rendered/K_test/calib.json')
    _same(calib_gt['K'], K, rtol=0.01)

    print('SUCCESS: test_calib_K_no_dist')


def test_calib_K_dist1():
    from calib_K import calc_intrinsics
    K, dist = calc_intrinsics('./data/calib_test_data/marker_32h11b2_4x4x_7cm.json',
                              'data/calib_test_data/rendered/K_dist_test/cam0/',
                              det_file_name=None,
                              estimate_dist=True,
                              dist_complexity=2)
    calib_gt = json_load('data/calib_test_data/rendered/K_dist_test/calib.json')
    _same(calib_gt['K'], K, rtol=0.01)
    _same(calib_gt['dist'], dist, rtol=0.05)

    print('SUCCESS: test_calib_K_dist1')


def test_calib_K_dist2():
    from calib_K import calc_intrinsics
    K, dist = calc_intrinsics('./data/calib_test_data/marker_32h11b2_4x4x_7cm.json',
                              'data/calib_test_data/rendered/K_dist_test2/cam0/',
                              det_file_name=None,
                              estimate_dist=True,
                              dist_complexity=2)
    calib_gt = json_load('data/calib_test_data/rendered/K_dist_test2/calib.json')
    _same(calib_gt['K'], K, rtol=0.01)
    _same(calib_gt['dist'], dist, rtol=0.05)

    print('SUCCESS: test_calib_K_dist2')


def test_calib_M():
    from calib_M import calc_extrinsics
    K, dist, M = calc_extrinsics('./data/calib_test_data/marker_32h11b2_4x4x_7cm.json',
                                 'data/calib_test_data/rendered/M_test/',
                                 'cam%d', 'run%03d', det_file_name=None, calib_file_name=None, calib_out_file_name=None,
                                 estimate_dist=False, dist_complexity=2,
                                 cache=False, verbose=0)

    calib_gt = json_load('data/calib_test_data/rendered/M_test/calib.json')
    for cid in range(len(K)):
        cam_name = 'cam%d' % cid
        _same(calib_gt['K'][cam_name], K[cid], rtol=0.01)
        _same(calib_gt['dist'][cam_name], dist[cid], rtol=0.05)
        _same(calib_gt['M'][cam_name], np.linalg.inv(M[cid]), atol=0.001, rtol=0.01)

    print('SUCCESS: test_calib_M')


def test_calib_M_dist():
    from calib_M import calc_extrinsics
    K, dist, M = calc_extrinsics('./data/calib_test_data/marker_32h11b2_4x4x_7cm.json',
                                 'data/calib_test_data/rendered/M_dist_test/',
                                 'cam%d', 'run%03d', det_file_name=None, calib_file_name=None, calib_out_file_name=None,
                                 estimate_dist=True, dist_complexity=1,
                                 optimize_distortion=False,
                                 cache=False, verbose=0)

    calib_gt = json_load('data/calib_test_data/rendered/M_dist_test/calib.json')
    for cid in range(len(K)):
        cam_name = 'cam%d' % cid
        _same(calib_gt['K'][cam_name], K[cid], rtol=0.01)
        _same(calib_gt['dist'][cam_name], dist[cid], atol=0.01, rtol=0.5)
        _same(calib_gt['M'][cam_name], np.linalg.inv(M[cid]), atol=0.01, rtol=0.01)

    print('SUCCESS: test_calib_M_dist')


if __name__ == '__main__':
    test_tag_detector(show=False)
    test_board_pose_estimator(show=False)
    test_calib_K_no_dist()
    test_calib_K_dist1()
    test_calib_K_dist2()
    test_calib_M()
    test_calib_M_dist()




