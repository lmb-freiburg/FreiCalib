import cv2
import argparse, os

from core.BoardDetector import BoardDetector
from utils.general_util import find_images, json_dump, json_load


def _detect_marker_video(marker_path, vid_data_path):
    # set up detector
    detector = BoardDetector(marker_path)

    # detect board in images
    points2d, point_ids = detector.process_video(vid_data_path)

    # image shape
    cap = cv2.VideoCapture(vid_data_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_shape = (h, w)
    return points2d, point_ids, img_shape, vid_data_path, os.path.dirname(vid_data_path)


def _detect_marker_img_folder(marker_path, img_data_path, verbose):
    # check for image files
    img_list = find_images(img_data_path)
    if verbose > 1:
        print('Found %s images for marker detection.' % len(img_list))

    # set up detector
    detector = BoardDetector(marker_path)

    # detect board in images
    points2d, point_ids = detector.process_image_batch(img_list)

    # image shape
    img_shape = cv2.imread(img_list[0]).shape[:2]

    files = [os.path.basename(x) for x in img_list]
    return points2d, point_ids, img_shape, files, img_data_path


def detect_marker(marker_path, data_path, output_file=None, cache=False, verbose=0):
    # check if folder/image or video case
    if os.path.isdir(data_path):
        # folder case
        base_dir = data_path
    else:
        # video case
        base_dir = os.path.dirname(data_path)

    # check for existing detection file
    if output_file is not None:
        det_file = os.path.join(base_dir, output_file)
        if cache and os.path.exists(det_file):
            if verbose > 0:
                print('Loading detection from: %s' % det_file)
            return json_load(det_file)

    if verbose > 0:
        print('Detection marker on:')
        print('\tData path: %s' % data_path)
        print('\tMarker file: %s' % marker_path)

    if os.path.isdir(data_path):
        if verbose > 0:
            print('\tAssuming: Folder of images.')
        points2d, point_ids, img_shape, files, base_dir = _detect_marker_img_folder(marker_path, data_path, verbose)

    else:
        if verbose > 0:
            print('\tAssuming: Video file.')
        points2d, point_ids, img_shape, files, base_dir = _detect_marker_video(marker_path, data_path)

    # save detections
    det = {'p2d': points2d,
           'pid': point_ids,
           'img_shape': img_shape,
           'files': files}

    if output_file is not None:
        json_dump(det_file, det, verbose=verbose > 0)

    return det


if __name__ == "__main__":
    """
        python detect_marker.py tags/marker_32h11b2_4x4x_7cm.json blender_scene/K_test/cam0/ -v2 --output_file detections.json
    """
    parser = argparse.ArgumentParser(description='Detect tags in images.')
    parser.add_argument('marker', type=str, help='Marker description file.')
    parser.add_argument('data_path', type=str, help='Path to where the recorded data is.')
    parser.add_argument('--output_file', type=str, default='detections.json', help='File to store detections in.'
                                                                                   ' If none is given doesnt save to disk.')
    parser.add_argument('-c', '--cache', action='store_true', help='Use stored version.')
    parser.add_argument('-v', '--verbosity', type=int, default=1, help='Verbosity level, higher is more ouput.')
    args = parser.parse_args()

    detect_marker(args.marker, args.data_path, args.output_file, args.cache, args.verbosity)
