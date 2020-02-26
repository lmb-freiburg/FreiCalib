#include <iostream>
#include <string>

#include "utils.h"
#include "InputReader.hpp"
#include "BundleAdjuster.hpp"

int main(int argc, char **argv) {    
    std::string inFile, outFile;    
    bool optimizeIntrinsic = false;
    bool optimizeRadial = false;
    bool optimizeExtrinsic = false;
    bool useSharedCameraModel = false;
    
    // Get command line arguments
    parseInputs(argc, argv,
                inFile, outFile, 
                optimizeIntrinsic, optimizeRadial, optimizeExtrinsic, useSharedCameraModel);

    
    // Read passed problem
    InputReader reader(inFile);
    reader.parse();
     
    // Setup problem
    BundleAdjuster adj(optimizeIntrinsic, optimizeRadial, optimizeExtrinsic, useSharedCameraModel,
                       reader.modelPoints3d,
                       reader.points2d, reader.points2dCamId, reader.points2dPtsId, reader.points2dFrameId, 
                       reader.cameraList,
                       reader.objectPoses);
    adj.optimize();
    adj.writeJson(outFile);
    
    return 0;
}
