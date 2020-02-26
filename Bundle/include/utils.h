#include <iostream>
#include <fstream>
#include <cmath>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

inline bool fileExistCheck(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void printProgramOptions(void) {
    std::cout << "FreiCalib Bundle adjuster, for optimizing an initial camera calibration.\n";
    std::cout << "Program options:\n";
    std::cout << "\t-i <string> : Input JSON describing the problem.\n";
    std::cout << "\t-o <string> : Ouput JSON after optimization.\n";
    
    std::cout << "\t-k : Optimize camera intrinsics (focals, principal point, [radials]).\n";
    std::cout << "\t-r : Optimize radial distortion.\n";
    std::cout << "\t-m : Optimize camera extrinsics (translation, rotation).\n";
    std::cout << "\t-s : Use a shared intrinsic camera model.\n";
}

int parseInputs(int argc, char **argv,
                std::string& inputFile, std::string& outputFile, 
                bool& optimizeIntrinsics, bool& optimizeDistortion, bool& optimizeExtrinsics, bool& shareCameraModel) {
    // Default values
    optimizeIntrinsics = false;
    optimizeDistortion = false;
    optimizeExtrinsics = false;
    shareCameraModel = false;
    
    int option;
    using namespace std;
    while ((option = getopt (argc, argv, "i:o:krms")) != -1) {
        switch (option)
            {       
            // INPUT DATA
            case 'i':
                inputFile = std::string(optarg);
                break;
            case 'o':
                outputFile = std::string(optarg);
                break;
                
            // FLAGS
            case 'k':
                optimizeIntrinsics = true;
                break;
            case 'r':
                optimizeDistortion = true;
                break;
            case 'm':
                optimizeExtrinsics = true;
                break;
            case 's':
                shareCameraModel = true;
                break;
                
            case '?':
                if (optopt == 'c')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
                fprintf (stderr,
                        "Unknown option character `\\x%x'.\n",
                        optopt);
                return 1;
            default:
                std::cout << "Something went wrong here\n";
                abort ();
            }
    }
    
//     std::cout << "\nProgram parameters:\n";
//     std::cout << "\tinputFile= " << inputFile << "\n";
//     std::cout << "\toutputFile= " << outputFile << "\n";
//     std::cout << "\toptimizeIntrinsics= " << optimizeIntrinsics << "\n";
//     std::cout << "\toptimizeDistortion= " << optimizeDistortion << "\n";
//     std::cout << "\toptimizeExtrinsics= " << optimizeExtrinsics << "\n";
//     std::cout << "\tshareCameraModel= " << shareCameraModel << "\n";
//     std::cout << "\n";
    
    if (inputFile.empty() || !fileExistCheck(inputFile)){
        std::cout << "\nMissing input file\n";
        printProgramOptions();
        exit(1);
        return 1;
    }
    if (outputFile.empty()){
        std::cout << "\nMissing output file\n";
        printProgramOptions();
        exit(1);
        return 1;
    }

//     for (index = optind; index < argc; index++)
//         printf ("Non-option argument %s\n", argv[index]);
    return 0;    
}
