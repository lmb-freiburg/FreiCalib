import argparse, os
import numpy as np
import cv2

from core.BoardDetector import BoardDetector
from utils.general_util import find_images, json_load


def show_marker_det(marker_path, data_path, det_file_name, block):
    print('Showing marker detections for::')
    print('\tData path: %s' % data_path)
    print('\tMarker file: %s' % marker_path)

    if os.path.isdir(data_path):
        print('\tAssuming: Folder of images.')
        _show_marker_det_img_folder(marker_path, data_path, det_file_name, block)

    else:
        print('\tAssuming: Video file.')
        _show_marker_det_video(marker_path, data_path, det_file_name, block)


def _show_marker_det_img_folder(marker_path, img_data_path, det_file_name, block):
    # load detections
    det_file = os.path.join(img_data_path, det_file_name)
    print('\tDetection file: %s' % det_file)
    assert os.path.exists(det_file), 'Could not find detection file.'
    det = json_load(det_file)

    # check for image files
    img_list = find_images(img_data_path)
    print('Found %s images for marker detection.' % len(img_list))

    # sanity check
    assert len(det['p2d']) == len(img_list), 'Number of detections and number of images differs.'
    assert len(det['pid']) == len(img_list), 'Number of detections and number of images differs.'

    # set up detector
    detector = BoardDetector(marker_path)

    # show
    for idx, img_p in enumerate(img_list):
        img_file = os.path.basename(img_p)
        if img_file not in det['files']:
            print('No detection available for: %s' % img_file)
            continue
        img = cv2.imread(img_p)
        detector.draw_board(img,
                            np.array(det['p2d'][idx]),
                            np.array(det['pid'][idx]),
                            block=block)


def _show_marker_det_video(marker_path, video_path, det_file_name, block):
    # load detections
    data_path = os.path.dirname(video_path)
    det_file = os.path.join(data_path, det_file_name)
    print('\tDetection file: %s' % det_file)
    assert os.path.exists(det_file), 'Could not find detection file.'
    det = json_load(det_file)

    # check video path
    video = cv2.VideoCapture(video_path)
    num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    print('Found video with %d frames.' % num_frames)

    # sanity check
    assert len(det['p2d']) == num_frames, 'Number of detections and number of frames differs.'
    assert len(det['pid']) == num_frames, 'Number of detections and number of frames differs.'

    # set up detector
    detector = BoardDetector(marker_path)

    # show
    idx = 0
    while True:
        if not video.isOpened():
            break
        ret, img = video.read()

        if not ret:
            break

        detector.draw_board(img,
                            np.array(det['p2d'][idx]),
                            np.array(det['pid'][idx]),
                            block=block)

        idx += 1


if __name__ == "__main__":
    """
        Shows calibration marker detections.
    """
    parser = argparse.ArgumentParser(description='Show marker detections.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data and the detection file is.')
    parser.add_argument('--block', action='store_true', help='If false automatically proceeds through all frames.')
    parser.add_argument('--det_file_name', type=str, default='detections.json', help='File detections are stored in.')
    args = parser.parse_args()

    show_marker_det(args.marker, args.data_path, args.det_file_name, args.block)
