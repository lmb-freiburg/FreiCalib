from __future__ import print_function, unicode_literals
import numpy as np

def _opencv_colormap(char):
    if char is None:
        color = [255, 255, 255]
    elif char == 'r':
        color = [255, 0, 0]
    elif char == 'g':
        color = [0, 255, 0]
    elif char == 'b':
        color = [0, 0, 255]
    elif char == 'w':
        color = [255, 255, 255]
    elif char == 'k':
        color = [0, 0, 0]
    else:
        color = [0, 0, 0]
    return color


def _default_camera(K=None, scale=1.0):
    fov_y, fov_x = 30.0 * np.pi / 180.0, 30.0 * np.pi / 180.0
    h, w = 1.0, 1.0
    if K is not None:
        fov_y = 2.0 * np.arctan(K[1, 2] / (K[1, 1]))
        fov_x = 2.0 * np.arctan(K[0, 2] / (K[0, 0]))
        h, w = 1.0, K[0, 1] / K[0, 2]

    l = 0.5 * h / np.tan(0.5 * fov_x)

    # define camera vertices: Camera points in z direction (plane is positive in z direction), centered in image
    vertices = np.array([[w / 2, h / 2, l],  # camera plane
                         [w / 2, -h / 2, l],
                         [-w / 2, -h / 2, l],
                         [-w / 2, h / 2, l],

                         [0.0, 0.0, 0.0],  # optic center
                         [w / 2 * 1.3, 0.0, l],  # x indicator point (is larger)
                         [0.0, h / 2 * 1.1, l]  # y indicator point
                         ])

    vertices *= scale

    edges = [[0, 1], [1, 2], [2, 3], [3, 0],  # camera plane
             [0, 4], [1, 4], [2, 4], [3, 4],  # lines from plane to center
             [0, 5], [1, 5],  # x indicator
             [0, 6], [3, 6]  # y indicator
             ]

    return vertices, edges


def dump_cameras_ply(file_out, camera_dict, is_world2cam=True, scale=0.1, cam_colors=None):
    """ Write camera calibration to disk for debugging. """
    for cam_name, M in camera_dict.items():
        if is_world2cam:
            M = np.linalg.inv(M)
        t = M[:3, 3].reshape([1, 3])
        R = M[:3, :3]

        vertices, edges = _default_camera(scale=scale)

        color = [0, 0, 0]  # default is black
        if cam_colors is not None:
            color = _opencv_colormap(cam_colors.get(cam_name, 'k'))

        # apply camera trans and rot
        vertices = np.matmul(vertices, R.T)
        vertices = vertices + t

        vertices = np.array(vertices).astype(np.float32)
        edges = np.array(edges).astype(np.int32)

        this_file_out = file_out + '_%s.ply' % cam_name
        with open(this_file_out, 'w') as fo:
            # header
            fo.write('ply\n')
            fo.write('format ascii 1.0\n')

            fo.write('element vertex %d\n' % len(vertices))
            fo.write('property float x\n')
            fo.write('property float y\n')
            fo.write('property float z\n')
            fo.write('property uchar red                   { start of vertex color }\n')
            fo.write('property uchar green\n')
            fo.write('property uchar blue\n')

            fo.write('element edge %d\n' % len(edges))
            fo.write('property int vertex1\n')
            fo.write('property int vertex2\n')
            fo.write('property uchar red                   { start of edge color }\n')
            fo.write('property uchar green\n')
            fo.write('property uchar blue\n')

            fo.write('end_header\n')

            # data
            for v in vertices:
                fo.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2],
                                                  color[0], color[1], color[2]))

            for i, j in edges:
                fo.write('%d %d %d %d %d\n' % (i, j,
                                               color[0], color[1], color[2]))
        print('Saved ply to: %s' % this_file_out)

def dump_point_array_ply(file_out, points, conf=None, threshold=None):
    """ Write a list of points with confidences to disk for debugging. """
    import matplotlib.pyplot as plt
    from colored import stylize, fg

    heat_map = plt.get_cmap('hot')
    if conf is not None:
        if threshold is not None:
            m = conf > threshold
            if np.sum(m) > 0:
                points = points[m, :]
                conf = conf[m]
                conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
                colors = heat_map(conf.flatten()) * 255
            else:
                print(stylize('WARNING:', fg('red')),
                      ' Array is empty because of threshold contraint:'
                      ' min=%.1f max=%.1f thresh=%.1f' % (conf.min(), conf.max(), threshold))
                return
        else:
            conf = (conf - conf.min()) / (conf.max() - conf.min() + 1e-8)
            colors = heat_map(conf.flatten()) * 255
    else:
        colors = (np.ones_like(points) * 255).astype(np.uint8)

    with open(file_out + '.ply', 'w') as fo:
        # header
        fo.write('ply\n')
        fo.write('format ascii 1.0\n')

        fo.write('element vertex %d\n' % points.shape[0])
        fo.write('property float x\n')
        fo.write('property float y\n')
        fo.write('property float z\n')
        fo.write('property uchar red                   { start of vertex color }\n')
        fo.write('property uchar green\n')
        fo.write('property uchar blue\n')

        fo.write('end_header\n')

        # data
        for c, v in zip(colors, points):
            fo.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))

    print('Saved ply to: %s' % file_out)


def dump_pcl_ply(file_out, points, colors):
    """ Write a list of points with colors to disk for debugging. """
    colors = (colors).astype(np.uint8)

    with open(file_out + '.ply', 'w') as fo:
        # header
        fo.write('ply\n')
        fo.write('format ascii 1.0\n')

        fo.write('element vertex %d\n' % points.shape[0])
        fo.write('property float x\n')
        fo.write('property float y\n')
        fo.write('property float z\n')
        fo.write('property uchar red                   { start of vertex color }\n')
        fo.write('property uchar green\n')
        fo.write('property uchar blue\n')

        fo.write('end_header\n')

        # data
        for c, v in zip(colors, points):
            fo.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))

    print('Saved ply to: %s' % file_out)
