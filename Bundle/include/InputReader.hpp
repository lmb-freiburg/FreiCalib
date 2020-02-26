#ifndef H_INPUTREADER
#define H_INPUTREADER

#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "json.hpp"

using namespace nlohmann;

class InputReader {
public:
    // Input file
    std::string m_inputFile;
    
    // Output data
    std::vector< std::vector<double> > modelPoints3d;
    std::vector< std::vector<double> > points2d;
    std::vector< unsigned int > points2dCamId;
    std::vector< unsigned int > points2dPtsId;
    std::vector< unsigned int > points2dFrameId;
    std::vector< std::vector<double> > cameraList;
    std::vector< std::vector<double> > objectPoses;
    
    InputReader(std::string inputFile):
        m_inputFile(inputFile) {
    }
    
    void parse(void) {
        // Stream file into json
        json fileDict;
        std::cout << "Reading input file: " << m_inputFile << "\n";
        std::ifstream in_file(m_inputFile);
        in_file >> fileDict;
        
        // Iterate keys of file
        for(json::iterator dataIter = fileDict.begin();
            dataIter != fileDict.end();
            ++dataIter) {
            
            if (dataIter.key() == "Camera") {
                this->_parse_cams(dataIter, cameraList);
//                 std::cout << "Num of cams: " << cameraList.size() << "\n";
            }
            
            else if (dataIter.key() == "ModelPoints") {
                this->_parse_mp(dataIter, modelPoints3d);
//                 std::cout << "Num of 3d model points: " << modelPoints3d.size() << "\n";
            }
            
            else if (dataIter.key() == "ObjectPoses") {
                this->_parse_objectPoses(dataIter, objectPoses);
//                 std::cout << "Num of object poses (num frames): " << objectPoses.size() << "\n";
            }
            
            else if (dataIter.key() == "ObservedPoints") {
                auto data = *dataIter;
                this->_parse_op_coords(data["coords"], points2d);
                this->_parse_op_id(data["pid"], points2dPtsId);
                this->_parse_op_id(data["cid"], points2dCamId);
                this->_parse_op_id(data["fid"], points2dFrameId);
//                 std::cout << "Num of observed points: " << points2d.size() << "\n";
            }
        }
    }
    
    // Parse camera into 
    void _parse_cams(json::iterator allCamDataIter,
                     std::vector< std::vector<double> >& cameraList) {
//         std::cout << "Parsing cameras\n";
        
        // Iter over all cameras
        for(json::iterator camIter = allCamDataIter->begin();
            camIter != allCamDataIter->end();
            ++camIter) {
            auto camData = *camIter;
        
            if (camData.size() != 17) {
                std::cout << "Camera needs to be defined by 15 parameters\n";
                std::cout.flush();
                exit(1);
            }
        
            // Iter data items of one camera
            std::vector<double> cam;
            for(json::iterator camDataIter = camData.begin();
                camDataIter != camData.end();
                ++camDataIter) {
                cam.push_back(*camDataIter);
            }
            cameraList.push_back(cam);
        }
    }
    
    // Parse model points
    void _parse_mp(json::iterator allWpDataIter,
                   std::vector< std::vector<double> >& points3d) {
//         std::cout << "Parsing model points\n";
        
        // Iter over all cameras
        for(json::iterator allWpData = allWpDataIter->begin();
            allWpData != allWpDataIter->end();
            ++allWpData) {
            auto wpData = *allWpData;
        
            if (wpData.size() != 3) {
                std::cout << "World point needs three scalars\n";
                std::cout.flush();
                exit(1);
            }
        
            // Iter data items of one camera
            std::vector<double> wp;
            for(json::iterator wpIter = wpData.begin();
                wpIter != wpData.end();
                ++wpIter) {
                wp.push_back(*wpIter);
            }
            points3d.push_back(wp);
        }
    }
    
    // Parse object poses
    void _parse_objectPoses(json::iterator allObjectPosesDataIter,
                   std::vector< std::vector<double> >& objectPoses) {
//         std::cout << "Parsing world points\n";
        
        // Iter over all cameras
        for(json::iterator allObjectPosesData = allObjectPosesDataIter->begin();
            allObjectPosesData != allObjectPosesDataIter->end();
            ++allObjectPosesData) {
            auto objectPosesData = *allObjectPosesData;
        
            if (objectPosesData.size() != 6) {
                std::cout << "Object pose needs six scalars\n";
                std::cout.flush();
                exit(1);
            }
        
            // Iter data items of one camera
            std::vector<double> op;
            for(json::iterator opIter = objectPosesData.begin();
                opIter != objectPosesData.end();
                ++opIter) {
                op.push_back(*opIter);
            }
            objectPoses.push_back(op);
        }
    }
    
    // Parse observed points
    void _parse_op_coords(json::array_t allOpData,
                          std::vector< std::vector<double> >& points2d) {
//         std::cout << "Parsing observed points\n";
        
        // Iter over all cameras
        for(unsigned int i=0; i < allOpData.size(); ++i) {        
            if (allOpData[i].size() != 2) {
                std::cout << "Observed point need two scalars\n";
                std::cout.flush();
                exit(1);
            }
        
            // Iter data items of one camera
            std::vector<double> op;
            for(unsigned int j=0; j < allOpData[i].size(); ++j) {
                op.push_back(allOpData[i][j]);
            }
            points2d.push_back(op);
        }
    }
    
    // Parse observed points camera/pt ids
    void _parse_op_id(json::array_t allOpData,
                       std::vector< unsigned int >& points2dId) {
//         std::cout << "Parsing observed points camera ids\n";
        
        // Iter over all cameras
        for(unsigned int i=0; i < allOpData.size(); ++i) {
            points2dId.push_back(allOpData[i]);
        }
    }
    
    
};

#endif