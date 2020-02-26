#ifndef H_REPROJECTIONERRORNOINTRINSIC
#define H_REPROJECTIONERRORNOINTRINSIC

#include "ceres/ceres.h"
#include "ceres/rotation.h"

#include "Camera.hpp"

// Templated pinhole camera model for Ceres.  
struct ReprojectionErrorNoIntrinsic {
    // Constructor
  ReprojectionErrorNoIntrinsic(double observed_x, double observed_y,
      double* cameraFocal, double* cameraPrincipal, double* cameraDist,
      double* modelPoint)
      : observed_x(observed_x), observed_y(observed_y),
      cameraFocal(cameraFocal), cameraPrincipal(cameraPrincipal), cameraDist(cameraDist),
      modelPoint(modelPoint) {}
      
    // Templated cost functor
  template <typename T>
  bool operator()(const T* const objectTranslation, // array of length 3
                  const T* const objectRotation, // array of length 3
                  const T* const cameraTranslation, // array of length 3
                  const T* const cameraRotation, // array of length 3
                  T* residuals) const {
                      
    // 3d model point
    T p[3]; 
    p[0] = (T) modelPoint[0];
    p[1] = (T) modelPoint[1];
    p[2] = (T) modelPoint[2];
    // Rotate model point according to the estimated object pose
    ceres::AngleAxisRotatePoint(objectRotation, p, p);
    
    // Translate the model point
    p[0] += objectTranslation[0];
    p[1] += objectTranslation[1];
    p[2] += objectTranslation[2]; 
    
    // Rotate point into the camera frame
    ceres::AngleAxisRotatePoint(cameraRotation, p, p);
    
    // cameraRotation[0, 1, 2] are the translation.
    p[0] += cameraTranslation[0];
    p[1] += cameraTranslation[1];
    p[2] += cameraTranslation[2];  //> Translate the point into the camera coordinate system
    
    // Compute the center of distortion.
    T xp = p[0] / p[2]; 
    T yp = p[1] / p[2];  
    
    // Camera distortion: OpenCV model with 5 parameters (3 radial, 2 tangential)
    double k1 = cameraDist[0];
    double k2 = cameraDist[1];
    double p1 = cameraDist[2];
    double p2 = cameraDist[3];
    double k3 = cameraDist[4]; 
    
    T r2 = xp*xp + yp*yp;
    T distortionRadial = 1.0 + r2*( k1 + r2*(k2 + r2*k3) );
    T distortionTangentialX = 2.0*p1*xp*yp + p2*(r2 + 2.0*xp*xp);
    T distortionTangentialY = 2.0*p2*xp*yp + p1*(r2 + 2.0*yp*yp);
    
    // Compute final projected point position.
    T predicted_x = cameraFocal[0] * (xp * distortionRadial + distortionTangentialX);
    T predicted_y = cameraFocal[1] * (yp * distortionRadial + distortionTangentialY);
    
    // Add principal point
    predicted_x = predicted_x + cameraPrincipal[0];
    predicted_y = predicted_y + cameraPrincipal[1];
    
    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;    
    return true;
  }
  
  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     double* cameraFocal,
                                     double* cameraPrincipal,
                                     double* cameraDist,
                                     double* modelPoint
                                    ) {
      return (new ceres::AutoDiffCostFunction<ReprojectionErrorNoIntrinsic, 2, 3, 3, 3, 3>( // Template argument is #output, #noparam1, #noparam2, ...
          new ReprojectionErrorNoIntrinsic(observed_x, observed_y, cameraFocal, cameraPrincipal, cameraDist, modelPoint)));
  }
  double observed_x;
  double observed_y;
  double* cameraFocal; // array of length 2
  double* cameraPrincipal; // array of length 2
  double* cameraDist; // array of length 5 (complete opencv distortion model)
  double* modelPoint;
};

#endif