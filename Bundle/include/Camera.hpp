#ifndef C_CAMERA
#define C_CAMERA

#include <Eigen/Dense>

// Data container for the camera parameters
class Camera {
public:
    double focal[2];  // focal lengths
    double principal[2]; // principal points
    double dist[5]; // parameters of the distortion
    double camRotation[3]; // Rodrigues vector (Camera rotation)
    double camTranslation[3]; // Camera translation
    int imgSize[2]; // height/width of image (for pp prior)
    
    Camera(double focalU, double focalV, 
           double principalU, double principalV, 
           double radial1, double radial2, double tang1, double tang2, double radial3, 
           double camRotation1, double camRotation2, double camRotation3, 
           double camTranslationX, double camTranslationY, double camTranslationZ,
           int width, int height
          ) {
        focal[0] = focalU;
        focal[1] = focalV;
        principal[0] = principalU;
        principal[1] = principalV;
        dist[0] = radial1;
        dist[1] = radial2;
        dist[2] = tang1;
        dist[3] = tang2;
        dist[4] = radial3;
        camRotation[0] = camRotation1;
        camRotation[1] = camRotation2;
        camRotation[2] = camRotation3;
        camTranslation[0] = camTranslationX;
        camTranslation[1] = camTranslationY;
        camTranslation[2] = camTranslationZ;
        imgSize[0] = width;
        imgSize[1] = height;
    }
    
    inline double* getFocal() { return this->focal; }
    inline double* getPrincipal() { return this->principal; }
    inline double* getDist() { return this->dist; }
    inline double* getRot() { return this->camRotation; }
    inline double* getTrans() { return this->camTranslation; }
};

#endif