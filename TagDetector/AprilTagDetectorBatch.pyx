# distutils: language = c++

from libc.string cimport memcpy
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

import numpy as np
cimport numpy as np # for np.ndarray

from libc.stdlib cimport malloc, free
from libc.string cimport strcmp
from cpython.string cimport PyString_AsString


def to_cstring(py_string):
    if isinstance(py_string, unicode):
        return py_string.encode('UTF-8')
    return py_string

cdef vector[string] to_cstring_array(list_str):
    cdef vector[string] ret
    cdef string tmp
    for i in xrange(len(list_str)):
        if isinstance(list_str[i], unicode):
            ret.push_back(list_str[i].encode('UTF-8'))
        else:
            ret.push_back(list_str[i])

    return ret

##########
""" DETECTION DATA CONTAINER"""

# Makes the cpp class available in here
cdef extern from "Detection.h":
    cdef cppclass Detection:
        Detection() except +
        Detection(string, int, vector[pair[float, float]]) except +
        string type
        int id
        vector[pair[float, float]] points

cdef class PyDetection:        
    cdef Detection* c_Detection      # holds pointer to an C++ instance which we're wrapping
    
    def __cinit__(self, type, int id, vector[pair[float, float]] points):
        
        # type is byte or unicode, but std::string wants bytes
        if isinstance(type, unicode):
            type = type.encode('UTF-8')
        
        self.c_Detection = new Detection(type, id, points)
        
    def __dealloc__(self):
        del self.c_Detection
    
    @property
    def type(self):
        return self.c_Detection.type.decode('UTF-8')

    @type.setter
    def type(self, value):
        if isinstance(value, unicode):
            value = value.encode('UTF-8')
        self.c_Detection.type = value
    
    @property
    def id(self):
        return self.c_Detection.id

    @id.setter
    def id(self, value):
        self.c_Detection.id = value
    
    @property
    def points(self):
        return self.c_Detection.points

    @points.setter
    def points(self, value):
        self.c_Detection.points = value

# Factory for creating the python equivalent of the c class
cdef object PyDetection_factory(Detection tmp):
    cdef string type_str
    
    if isinstance(tmp.type, unicode):
        type_str = tmp.type.encode('UTF-8')
    else:
        type_str = tmp.type
    cdef PyDetection py_obj = PyDetection(type_str, tmp.id, tmp.points)
    return py_obj
        

###########
#""" APRILTAG DETECTOR BATCH"""
    
cdef extern from "RunAprilDetectorBatch.hpp":
    # Declares that we want to use this class here
    cdef cppclass RunAprilDetectorBatch:
        # list attributes and methods we are going to use
        RunAprilDetectorBatch(string) except +  # this is just the constructor; weird stuff turns cpp exceptions into python exceptions
        RunAprilDetectorBatch(string, bool) except +  # this is just the constructor; weird stuff turns cpp exceptions into python exceptions
        RunAprilDetectorBatch(string, int, unsigned int, bool, float) except +  # this is just the constructor; weird stuff turns cpp exceptions into python exceptions
        vector[vector[Detection]] processImageBatch(vector[string])
        vector[vector[Detection]] processVideo(string)


cdef class PyRunAprilDetectorBatch:
    cdef RunAprilDetectorBatch* c_RunAprilDetectorBatch      # holds pointer to an C++ instance which we're wrapping
    def __cinit__(self, codeName, int blackBorder, unsigned int maxNumThreads, float resizeFactor, bool draw=False):
        
        if isinstance(codeName, unicode):
            codeName = codeName.encode('UTF-8')
        self.c_RunAprilDetectorBatch = new RunAprilDetectorBatch(codeName, blackBorder, maxNumThreads, draw, resizeFactor)
        
    def __dealloc__(self):
        del self.c_RunAprilDetectorBatch
        
    def processImageBatch(self, imagePaths):
        cdef vector[string] imagePathsEnc = to_cstring_array(imagePaths)
        # imagePathsEnc = list()
        # for x in imagePaths:
        #    imagePathsEnc.append(x.encode('UTF-8'))

        # cdef vector[string] imagePathsEnc
        # for x in imagePaths:
        #     imagePathsEnc.push_back(x.encode('UTF-8'))

        # get the c class result
        cdef vector[vector[Detection]] cResult = self.c_RunAprilDetectorBatch.processImageBatch(imagePathsEnc)

        fullOut = list()
        for imgResult in cResult:
            imgOut = list()
            for x in imgResult:
                imgOut.append(PyDetection_factory(x))
            fullOut.append(imgOut)
        
        return fullOut

    def processVideo(self, videoPath):
        cdef string videoPathStr = to_cstring(videoPath)

        # get the c class result
        cdef vector[vector[Detection]] cResult = self.c_RunAprilDetectorBatch.processVideo(videoPathStr)

        fullOut = list()
        for imgResult in cResult:
            imgOut = list()
            for x in imgResult:
                imgOut.append(PyDetection_factory(x))
            fullOut.append(imgOut)

        return fullOut