import os
from distutils.core import setup, Extension
from Cython.Build import cythonize

# source files of the apriltag detector
CPP_FILES =['./library/src/FloatImage.cc', './library/src/TagDetection.cc', './library/src/TagDetector.cc',
            './library/src/TagFamily.cc', './library/src/Edge.cc', './library/src/Gaussian.cc', './library/src/GLine2D.cc',
            './library/src/GLineSegment2D.cc', './library/src/GrayModel.cc',  './library/src/Homography33.cc',
            './library/src/MathUtil.cc', './library/src/Quad.cc', './library/src/Segment.cc', './library/src/UnionFindSimple.cc']

CPP_FILES.append('AprilTagDetectorBatch.pyx')
ext = Extension('AprilTagDetectorBatch',
                sources=[os.path.abspath(x) for x in CPP_FILES],
                libraries=['opencv_highgui', 'opencv_core', 'opencv_calib3d', 'opencv_video', 'opencv_videoio', 'pthread'],
                include_dirs=[os.path.abspath('./library/include/')],
                extra_compile_args=['-std=c++11']
                )

setup(
    name = "AprilTagDetector",
    ext_modules = cythonize(ext),
)

#, 'opencv_videostab', 'opencv_video', 'opencv_ts', 'opencv_superres', 'opencv_stitching', 'opencv_photo', 'opencv_ocl', 'opencv_objdetect', 'opencv_ml', 'opencv_legacy', 'opencv_imgproc', 'opencv_gpu', 'opencv_flann', 'opencv_features2d',  'opencv_contrib', 'opencv_calib3d'