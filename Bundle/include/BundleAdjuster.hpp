#ifndef H_BUNDLEADJUSTER
#define H_BUNDLEADJUSTER

#include <array>
#include <vector>
#include <list>
#include <fstream>
#include <iostream>

#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "json.hpp"

#include "ReprojectionErrorWithRadialFull.h"
#include "ReprojectionError.h"
#include "ReprojectionErrorNoIntrinsic.h"
#include "ReprojectionErrorNoIntrinsicNoExtrinsic.h"
#include "Camera.hpp"

using namespace nlohmann;

class ObservedPoint {
public:
    unsigned int cid;
    unsigned int pid;
    unsigned int fid;
    double u;
    double v;
    
    ObservedPoint(unsigned int cid, unsigned int pid, unsigned int fid,
                  double u, double v):
        cid(cid),
        pid(pid),
        fid(fid),
        u(u),
        v(v)
        { }
};

class WorldPoint {
public:
    double coord[3];
    
    WorldPoint(double x, double y, double z) { 
            coord[0] = x;
            coord[1] = y;
            coord[2] = z;
        }
        
    double* get(void){
        return this->coord;
    }
};

class BundleAdjuster {
private:
    bool m_optimizeIntrinsic;
    bool m_optimizeRadial;
    bool m_optimizeExtrinsic;
    bool m_useSharedCameraModel;
    
    double* m_modelPoints3d;
    double* m_objectPoses;
    unsigned int m_numFrames;
    unsigned int m_num3dPoints;
    std::vector< ObservedPoint > m_points2d;
    std::vector<Camera> m_camera;
    
public:
    BundleAdjuster(bool optimizeIntrinsic, bool optimizeRadial, bool optimizeExtrinsic, bool useSharedCameraModel,
                   std::vector< std::vector<double> >& modelPoints3d,
                   std::vector< std::vector<double> >& points2d,
                   std::vector< unsigned int >& points2dCamId,
                   std::vector< unsigned int >& points2dPtsId,
                   std::vector< unsigned int >& points2dFrameId,
                   std::vector< std::vector<double> >& cameraList,
                   std::vector< std::vector<double> >& objectPosesList):
            m_optimizeIntrinsic(optimizeIntrinsic),
            m_optimizeRadial(optimizeRadial),
            m_optimizeExtrinsic(optimizeExtrinsic),
            m_useSharedCameraModel(useSharedCameraModel){
        
        // Copy 3d points
        m_num3dPoints = modelPoints3d.size();
        m_modelPoints3d = new double[m_num3dPoints*3];
        for (unsigned int i=0; i< m_num3dPoints; ++i) {
            m_modelPoints3d[i*3 + 0] = modelPoints3d[i][0];
            m_modelPoints3d[i*3 + 1] = modelPoints3d[i][1];
            m_modelPoints3d[i*3 + 2] = modelPoints3d[i][2];
        }
        
        // Copy observed points
        for (unsigned int i=0; i< points2d.size(); ++i) {
            m_points2d.push_back(ObservedPoint(points2dCamId[i],
                                               points2dPtsId[i],
                                               points2dFrameId[i],
                                               points2d[i][0],
                                               points2d[i][1]));
        }
        
        // Copy cameras
        for (unsigned int i=0; i< cameraList.size(); ++i) {
            m_camera.push_back(Camera(cameraList[i][0], cameraList[i][1],
                                      cameraList[i][2], cameraList[i][3],
                                      cameraList[i][4], cameraList[i][5], cameraList[i][6], cameraList[i][7], cameraList[i][8],
                                      cameraList[i][9], cameraList[i][10], cameraList[i][11],
                                      cameraList[i][12], cameraList[i][13], cameraList[i][14],
                                      cameraList[i][15], cameraList[i][16]));
        }
        
        // Copy object poses
        m_numFrames = objectPosesList.size();
        m_objectPoses = new double[m_numFrames*6];
        for (unsigned int i=0; i< objectPosesList.size(); ++i)
        for (unsigned int j=0; j< 6; ++j) {
            m_objectPoses[i*6 + j] = objectPosesList[i][j];
        }

    }
    
    ~BundleAdjuster() {
        delete [] m_modelPoints3d;
        delete [] m_objectPoses;
    }
    
    void printCameras(void) {
        std::cout << "----------------------------\n";
        std::cout << "CAMERAS\n";
        
        for (unsigned int i=0; i<m_camera.size(); ++i) {
            std::cout << "\tCAM " << i << "\n";
            std::cout << "\tFocals= " << m_camera[i].focal[0] << " / " << m_camera[i].focal[1] << "\n";
            std::cout << "\tPrincipals= " << m_camera[i].principal[0] << " / " << m_camera[i].principal[1] << "\n";
            std::cout << "\tRadials= " << m_camera[i].dist[0] << " / " << m_camera[i].dist[1] <<  " / " << m_camera[i].dist[2] <<  " / " << m_camera[i].dist[3] <<  " / " << m_camera[i].dist[4] << "\n";
            std::cout << "\tCam rotation= " << m_camera[i].camRotation[0] << " / " << m_camera[i].camRotation[1] << " / " << m_camera[i].camRotation[2] << "\n";
            std::cout << "\tCam translation= " << m_camera[i].camTranslation[0] << " / " << m_camera[i].camTranslation[1] << " / " << m_camera[i].camTranslation[2] << "\n";
            std::cout << "\t-------------\n";
        }
        std::cout << "----------------------------\n";
        std::cout.flush();
    }
    
    void writeJson(std::string outputFilePath) {     
        std::ofstream outputFile(outputFilePath);
        json outputDict;
        
        // Write cameras
        for (unsigned int i=0; i<m_camera.size(); ++i) {
            unsigned int cid = i;
            if (m_useSharedCameraModel) {
                // In case of a shared model only the first cameras focal/pp/dist was optimized
                cid = 0;
            }
            outputDict["Camera"][i] = {m_camera[cid].focal[0], m_camera[cid].focal[1],
                                       m_camera[cid].principal[0], m_camera[cid].principal[1],
                                       m_camera[cid].dist[0], m_camera[cid].dist[1], m_camera[cid].dist[2], m_camera[cid].dist[3], m_camera[cid].dist[4],
                                       m_camera[i].camRotation[0], m_camera[i].camRotation[1], m_camera[i].camRotation[2],
                                       m_camera[i].camTranslation[0], m_camera[i].camTranslation[1], m_camera[i].camTranslation[2]};
        }
        
        
        // Write ObjectPoses
        std::vector< std::vector<double> > objectPosesList;
        for (unsigned int i=0; i < m_numFrames; ++i){
            objectPosesList.push_back(  { m_objectPoses[i*6],
                                          m_objectPoses[i*6+1],
                                          m_objectPoses[i*6+2],
                                          m_objectPoses[i*6+3],
                                          m_objectPoses[i*6+4],
                                          m_objectPoses[i*6+5]});
        }
        outputDict["ObjectPoses"] = objectPosesList;
        
        std::cout << "Wrote output file: " << outputFilePath << "\n";
        outputFile << std::setw(4) << outputDict << std::endl;
        std::cout.flush();
    }
    
    void optimize(void) {                        
        // Set up optimization problem
        ceres::Problem problem;
        
        if (m_optimizeRadial && m_optimizeIntrinsic && m_optimizeExtrinsic) {
            std::cout << "Optimizing object pose, camera intrinsics, distortion and camera extrinsics.\n";
        } 
        else if (!m_optimizeRadial && m_optimizeIntrinsic && m_optimizeExtrinsic) {
            std::cout << "Optimizing object pose, camera intrinsics and extrinsics.\n";
        }
        else if (!m_optimizeIntrinsic && m_optimizeExtrinsic) {
            std::cout << "Optimizing object pose and camera extrinsics\n";
        }
        else if (!m_optimizeRadial && !m_optimizeIntrinsic && !m_optimizeExtrinsic) {
            std::cout << "Optimizing only object pose.\n";
        }
        
        for (int i = 0; i < m_points2d.size(); ++i) {
            unsigned int cid = m_points2d[i].cid;  // camera id
            unsigned int cidShared = m_points2d[i].cid;  // camera id indicating where the shared parameters are stored
            unsigned int pid = m_points2d[i].pid;  // point id
            unsigned int fid = m_points2d[i].fid;  // frame id
            
            if (m_useSharedCameraModel) {
                // In case of the shared camera model we only optimize the first set
                cidShared = 0;
            }
            
            // Each Residual block takes a point and a camera as input and outputs a 2
            // dimensional residual. Internally, the cost function stores the observed
            // image location and compares the reprojection against the observation.
            if (m_optimizeRadial && m_optimizeIntrinsic && m_optimizeExtrinsic) {
//                 std::cout << "Optimizing 3D points, camera intrinsics, distortion and camera extrinsics.\n";
                ceres::CostFunction* cost_function = ReprojectionErrorWithRadialFull::Create(m_points2d[i].u,
                                                                                             m_points2d[i].v,
                                                                                             &m_camera[cid].imgSize[0],
                                                                                             &m_camera[cid].imgSize[1],
                                                                                             &m_modelPoints3d[pid*3]);
                
                ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);
                problem.AddResidualBlock(cost_function,
    //                                         loss_function, // NULL /* squared loss */,
                                        NULL /* squared loss */,
                                        &m_objectPoses[fid*6 + 3],  // 6 params per frame; second 3 translation
                                        &m_objectPoses[fid*6], // first 3 rotation
                                        m_camera[cid].getTrans(),
                                        m_camera[cid].getRot(),
                                        m_camera[cidShared].getFocal(),
                                        m_camera[cidShared].getPrincipal(),
                                        m_camera[cidShared].getDist());
            } 
            else if (!m_optimizeRadial && m_optimizeIntrinsic && m_optimizeExtrinsic) {
//                 std::cout << "Optimizing 3D points, camera intrinsics and extrinsics.\n";
                ceres::CostFunction* cost_function = ReprojectionError::Create(m_points2d[i].u,
                                                                               m_points2d[i].v,
                                                                               &m_camera[cid].imgSize[0],
                                                                               &m_camera[cid].imgSize[1],
                                                                               m_camera[cidShared].getDist(),
                                                                               &m_modelPoints3d[pid*3]);
                
                ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);
                problem.AddResidualBlock(cost_function,
//                                         loss_function, // NULL /* squared loss */,
                                        NULL /* squared loss */,
                                        &m_objectPoses[fid*6 + 3],  // 6 params per frame; second 3 translation
                                        &m_objectPoses[fid*6], // first 3 rotation
                                        m_camera[cid].getTrans(),
                                        m_camera[cid].getRot(),
                                        m_camera[cidShared].getFocal(),
                                        m_camera[cidShared].getPrincipal());
            }
            else if (!m_optimizeIntrinsic && m_optimizeExtrinsic) {
//                 std::cout << "Optimizing 3D points and camera extrinsics\n";
                ceres::CostFunction* cost_function = ReprojectionErrorNoIntrinsic::Create(m_points2d[i].u,
                                                                                          m_points2d[i].v,
                                                                                          m_camera[cidShared].getFocal(),
                                                                                          m_camera[cidShared].getPrincipal(),
                                                                                          m_camera[cidShared].getDist(), &m_modelPoints3d[pid*3]);
                
                ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);
                problem.AddResidualBlock(cost_function,
//                                         loss_function, // NULL /* squared loss */,
                                        NULL /* squared loss */,
                                        &m_objectPoses[fid*6 + 3],  // 6 params per frame; second 3 translation
                                        &m_objectPoses[fid*6], // first 3 rotation
                                        m_camera[cid].getTrans(),
                                        m_camera[cid].getRot());
            }
            else if (!m_optimizeRadial && !m_optimizeIntrinsic && !m_optimizeExtrinsic) {
//                 std::cout << "Optimizing only the 3D points\n";
                ceres::CostFunction* cost_function = ReprojectionErrorNoIntrinsicNoExtrinsic::Create(m_points2d[i].u,
                                                                                          m_points2d[i].v,
                                                                                          m_camera[cid].getTrans(),
                                                                                          m_camera[cid].getRot(),
                                                                                          m_camera[cidShared].getFocal(),
                                                                                          m_camera[cidShared].getPrincipal(),
                                                                                          m_camera[cidShared].getDist(), &m_modelPoints3d[pid*3]);
                
                ceres::LossFunction* loss_function = new ceres::HuberLoss(0.2);
                problem.AddResidualBlock(cost_function,
//                                         loss_function, // NULL /* squared loss */,
                                        NULL /* squared loss */,
                                        &m_objectPoses[fid*6 + 3],  // 6 params per frame; second 3 translation
                                        &m_objectPoses[fid*6]); // first 3 rotation
                                        
            }
        }
        
        std::cout << "Problem defined. Starting optimization...\n";
        std::cout.flush();
        
        // Make Ceres automatically detect the bundle structure. Note that the
        // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
        // for standard bundle adjustment problems.
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;  // DENSE_SCHUR ideal for up to a hundred variables
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 50;
        options.function_tolerance = 1e-4;  // minium percentage of cost change
//         options.initial_trust_region_radius = 1e2;  
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
//         std::cout << summary.FullReport() << "\n";
    }
    
    
};

#endif