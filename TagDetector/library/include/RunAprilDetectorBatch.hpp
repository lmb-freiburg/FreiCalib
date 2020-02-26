#ifndef RUNAPRILDETECTORBATCH_H
#define RUNAPRILDETECTORBATCH_H

#include <string.h>
#include <vector>
#include <thread>
#include <mutex>
#include <queue>
#include <utility>

#include "opencv2/opencv.hpp"

// April tags detector and various families that can be selected by command line option
#include "TagDetector.h"
#include "Tag16h5.h"
#include "Tag36h11.h"

#include "Detection.h"

class RunAprilDetectorBatch {
private:
    AprilTags::TagDetector* m_tagDetector;
    AprilTags::TagCodes m_tagCodes;
    std::string m_tagCodesName;
    bool m_stop;  // Stopping flag for video workers
    bool m_draw; // Indicates if detections should be showed or not 
    int m_blackBorder;  // Amount of border 
    unsigned int m_maxNumThreads; // Number of parallel threads
    
    cv::Mat m_image;  // Image read from disk
    cv::Mat m_image_gray;  // Grayscale image for detection
    
    float m_resizeFactor;
    
public:
    // default constructor
    RunAprilDetectorBatch(std::string codeName, int blackBorder):
        // default settings
        m_tagDetector(NULL),
        m_maxNumThreads(1),
        m_resizeFactor(1.0),
        m_tagCodes(AprilTags::tagCodes36h11),
        m_blackBorder(blackBorder),
        m_draw(false),
        m_stop(false)
        {
            // Set the tag family
            if (codeName == "16h5") {
                m_tagCodes = AprilTags::tagCodes16h5;
            }
            else if (codeName == "36h11") {
                m_tagCodes = AprilTags::tagCodes36h11;
            }
            else {
                std::cout << "Unknown Tag family:" << codeName << "\n";
                std::cout << "Using default: 36h11b1\n";
                m_tagCodes = AprilTags::tagCodes36h11;
                m_blackBorder = 1;
            }
            m_tagCodesName = codeName + std::to_string(m_blackBorder);
            
            // Set up detector
            setup();
        }
        
    RunAprilDetectorBatch(std::string codeName, int blackBorder, unsigned int maxNumThreads, bool draw, float resizeFactor):
        // default settings
        m_tagDetector(NULL),
        m_maxNumThreads(maxNumThreads),
        m_resizeFactor(resizeFactor),
        m_tagCodes(AprilTags::tagCodes36h11),
        m_blackBorder(blackBorder),
        m_draw(draw),
        m_stop(false)
        {
            // Set the tag family
            if (codeName == "16h5") {
                m_tagCodes = AprilTags::tagCodes16h5;
            }
            else if (codeName == "36h11") {
                m_tagCodes = AprilTags::tagCodes36h11;
            }
            else {
                std::cout << "Unknown Tag family:" << codeName << "\n";
                std::cout << "Using default: 36h11b1\n";
                m_tagCodes = AprilTags::tagCodes36h11;
                m_blackBorder = 1;
            }
            m_tagCodesName = codeName + std::to_string(m_blackBorder);
            
            // Set up detector
            setup();
        }
        
    // call this once to create a TagDetector
    void setup() {
        if (m_tagDetector != NULL) {
            delete m_tagDetector;
            cv::destroyWindow("apriltag_det");
        }

        m_tagDetector = new AprilTags::TagDetector(m_tagCodes, m_blackBorder);

        // prepare window for drawing the camera images
        if (m_draw) {
            cv::namedWindow("apriltag_det", 1);
        }
    }
    
    
    std::vector< std::vector< Detection > > processImageBatch(std::vector<std::string> imagePathBatch) {
        std::vector< std::vector< Detection > > detectionResult;  // This is where we keep the output
        detectionResult.resize(imagePathBatch.size()); // Thats how much output we will have
        std::vector< int > processIdList;
        
        // Fill vector of process ids
        for (int pid=0; pid < static_cast<unsigned int> (imagePathBatch.size()); ++pid) {
            processIdList.push_back(pid);
        }
        
        std::vector< std::thread > workerList;  // Keep track of our workers
        unsigned int maxNumThreads = std::min(m_maxNumThreads, static_cast<unsigned int> (imagePathBatch.size()));  // Possible that there are less jobs than possible threads
//         std::cout << "Running with " << maxNumThreads << " threads\n";       
        
        std::mutex getJobMutex, writeResultMutex;
        
        // Create worker threads
        for(unsigned int i=0; i < maxNumThreads; i++) {
//             std::cout << "Starting thread " << i << "\n";            
            workerList.push_back(std::thread(&RunAprilDetectorBatch::processImageWorkerThread, this,
                                             std::ref(processIdList),
                                             std::ref(imagePathBatch),
                                             std::ref(detectionResult),
                                             std::ref(getJobMutex),
                                             std::ref(writeResultMutex))
                       );
        }
        
//         std::cout << "Now waiting for jobs to finish. \n";   
        
        // Let all workers finish
        while (workerList.size() > 0) {                
            
            // Iterate workers and check if one of them is finished, if so remove from the list of workers
            for (unsigned int j=0; j < workerList.size(); ++j) {
                if (workerList[j].joinable()) {
                    workerList[j].join();
                    workerList.erase(workerList.begin() + j);
//                     std::cout << "Some Thread finished and joined\n";
                    break;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        
        return detectionResult;
        
    }
    
    
    void processImageWorkerThread(std::vector<int>& processIdList, std::vector< std::string>& imagePathList,
                                  std::vector< std::vector< Detection > >& detectionResult,
                                  std::mutex& getJobMutex, std::mutex& writeResultMutex) {
        // Dont use class members, because they are shared across threads
        cv::Mat image;  // Image read from disk
        cv::Mat image_small;  // Image resized
        cv::Mat image_gray;  // Grayscale image for detection
        
        // Input data of a single job
        int processId;
        std::string imagePath;
        
        while (true) { // worker loop (loops until it breaks, which happens when there are no more jobs)
            
            // Check for a work package
            getJobMutex.lock();
            if (processIdList.size() == 0) {
//                 std::cout << "Stopping worker thread.\n";
                getJobMutex.unlock();
                break;
            }
            else {
                // Get data to do a job
                processId = processIdList.back();
                imagePath = imagePathList.back();
                processIdList.pop_back();
                imagePathList.pop_back();
//                 std::cout << "Took job " << processId << "\n";
                getJobMutex.unlock();
            }
            
            //// Actually do the job
            // Read image
            image = cv::imread(imagePath);
            cv::resize(image, image_small, cv::Size(), m_resizeFactor, m_resizeFactor);
            
            // detect April tags (requires a gray scale image)
            cv::cvtColor(image_small, image_gray, CV_BGR2GRAY);
            vector<AprilTags::TagDetection> detections = m_tagDetector->extractTags(image_gray);

            // show the current image including any detections
            if (m_draw) {
                for (unsigned int i=0; i<detections.size(); i++) {
                    // also highlight in the image
                    detections[i].draw(image);
                }
                char s[255];
                sprintf(s, "apriltag_det%d", processId);
                cv::imshow(s, image); // OpenCV call
                cv::waitKey(200);
            }
            
            writeResultMutex.lock();
            float f = 1.0f / m_resizeFactor; // upscaleFactor
            // Put everything in a generic detection
            for (unsigned int i=0; i<detections.size(); i++) {
                std::vector< std::pair<float, float> > points;
                points.push_back(std::pair<float, float> (detections[i].p[0].first*f, detections[i].p[0].second*f));
                points.push_back(std::pair<float, float> (detections[i].p[1].first*f, detections[i].p[1].second*f));
                points.push_back(std::pair<float, float> (detections[i].p[2].first*f, detections[i].p[2].second*f));
                points.push_back(std::pair<float, float> (detections[i].p[3].first*f, detections[i].p[3].second*f));
//                 points.push_back(detections[i].p[0]);
//                 points.push_back(detections[i].p[1]);
//                 points.push_back(detections[i].p[2]);
//                 points.push_back(detections[i].p[3]);
                detectionResult[processId].push_back(Detection(this->m_tagCodesName,
                                                    detections[i].id,
                                                    points));
            }      
//             std::cout << "Finished job " << processId << " and wrote back results\n";
            writeResultMutex.unlock();
            
        } // worker loop
    }
    
    std::vector< Detection > processImage(std::string imagePath) {
        std::vector< Detection > detectionResult;
        
        // Read image
        m_image = cv::imread(imagePath);
        
        // detect April tags (requires a gray scale image)
        cv::cvtColor(m_image, m_image_gray, CV_BGR2GRAY);
        vector<AprilTags::TagDetection> detections = m_tagDetector->extractTags(m_image_gray);

//         // print out each detection
//         cout << detections.size() << " tags detected:" << endl;
//         for (unsigned int i=0; i<detections.size(); i++) {
//             print_detection(detections[i]);
//         }

        // show the current image including any detections
        if (m_draw) {
            for (unsigned int i=0; i<detections.size(); i++) {
                // also highlight in the image
                detections[i].draw(m_image);
            }
            cv::imshow("apriltag_det", m_image); // OpenCV call
        }
        
        // Put everything in a generic detection
        for (unsigned int i=0; i<detections.size(); i++) {
            std::vector< std::pair<float, float> > points;
            points.push_back(detections[i].p[0]);
            points.push_back(detections[i].p[1]);
            points.push_back(detections[i].p[2]);
            points.push_back(detections[i].p[3]);
            detectionResult.push_back(Detection(this->m_tagCodesName,
                                                detections[i].id,
                                                points));
        }
        return detectionResult;
        
    }
    
    void print_detection(AprilTags::TagDetection& detection) const {
        std::cout << "  Id: " << detection.id << " (Hamming: " << detection.hammingDistance << ")";
    }


    std::vector< std::vector< Detection > > processVideo(std::string videoPath) {
        // Open Video
        cv::VideoCapture video(videoPath);
        unsigned int numOfFrames(video.get(CV_CAP_PROP_FRAME_COUNT));

        std::vector< std::vector< Detection > > detectionResult;  // This is where we keep the output
        detectionResult.resize(numOfFrames); // Thats how much output we will have

        std::vector< std::thread > workerList;  // Keep track of our workers
        std::queue< std::pair<unsigned int, cv::Mat> > frameQueue; // Queue of read images
        std::mutex queueLock, writeResultMutex;

        // Start Worker that reads new frames from the video
        std::thread videoReader(&RunAprilDetectorBatch::videoReaderThread, this,
                               std::ref(video),
                               std::ref(queueLock),
                               std::ref(frameQueue));

        // Use remaining workers for making detections
        m_stop = false;
        for(unsigned int i=0; i < m_maxNumThreads; i++) {
            workerList.push_back(std::thread(&RunAprilDetectorBatch::processVideoWorkerThread, this,
                                             std::ref(detectionResult),
                                             std::ref(queueLock),
                                             std::ref(frameQueue),
                                             std::ref(writeResultMutex)));
        }


        // Let all workers finish
        for (unsigned int j=0; j < workerList.size(); ++j) {
            workerList[j].join();
        }

        // clean up remainder
        videoReader.join();
        video.release();

        return detectionResult;

    }

    void videoReaderThread(cv::VideoCapture& video, std::mutex& queueLock, std::queue< std::pair<unsigned int, cv::Mat> >& frameQueue){

//        std::cout << "Video reader thread created\n";
        // Check if camera opened successfully
        if(!video.isOpened()){
//            std::cout << "Error opening video stream or file\n";
            return;
        }

        cv::Mat frame;
        unsigned int fid=0;
        while (true) {
            video >> frame;
            if (frame.empty()){
                m_stop = true;  // this will tell the workers that they can stop once the queue is empty
                return;
            }

            // wait for lock
            queueLock.lock();

            // Once we have it we add something to the queue
            frameQueue.push(
                std::make_pair(fid, frame.clone())
            );
            fid++;
            queueLock.unlock();

//            std::cout << "Number of frames in queue: " << frameQueue.size() << "\n";

            // We dont fill the queue too much
            while (frameQueue.size() > 32){
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }
    }

    void processVideoWorkerThread(std::vector< std::vector< Detection > >& detectionResult,
                                  std::mutex& queueLock,
                                  std::queue< std::pair<unsigned int, cv::Mat> >& frameQueue, std::mutex& writeResultMutex){
            // Read image
            cv::Mat image, image_gray, image_small;
            unsigned int fid;

            // main worker loop
            while (true){

                // wait for frames loop
                while (true){
                    queueLock.lock();
                    if (frameQueue.empty()) {
                        if (m_stop) {
                            // end criterion: empty queue with stop signal
                            queueLock.unlock();
                            return;
                        }
                        else {
                            // wait for more frames
                            queueLock.unlock();
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }

                    } else{
                        // process given frame
                        std::swap(frameQueue.front().first, fid);
                        std::swap(frameQueue.front().second, image);
                        frameQueue.pop();
                        queueLock.unlock();
                        break;
                    }

                }

                // resize
                cv::resize(image, image_small, cv::Size(), m_resizeFactor, m_resizeFactor);

                // detect April tags (requires a gray scale image)
                cv::cvtColor(image_small, image_gray, CV_BGR2GRAY);
                vector<AprilTags::TagDetection> detections = m_tagDetector->extractTags(image_gray);
//                std::cout << "Detection done.\n";

                // show the current image including any detections
                if (m_draw) {
                    for (unsigned int i=0; i<detections.size(); i++) {
                        // also highlight in the image
                        detections[i].draw(image);
                    }
                    char s[255];
                    sprintf(s, "apriltag_det");
                    cv::imshow(s, image); // OpenCV call
                    cv::waitKey(100);

                    cv::destroyWindow("apriltag_det");
                }

                writeResultMutex.lock();
                float f = 1.0f / m_resizeFactor; // upscaleFactor
                // Put everything in a generic detection
                for (unsigned int i=0; i<detections.size(); i++) {
                    std::vector< std::pair<float, float> > points;
                    points.push_back(std::pair<float, float> (detections[i].p[0].first*f, detections[i].p[0].second*f));
                    points.push_back(std::pair<float, float> (detections[i].p[1].first*f, detections[i].p[1].second*f));
                    points.push_back(std::pair<float, float> (detections[i].p[2].first*f, detections[i].p[2].second*f));
                    points.push_back(std::pair<float, float> (detections[i].p[3].first*f, detections[i].p[3].second*f));
                    detectionResult[fid].push_back(Detection(this->m_tagCodesName,
                                                        detections[i].id,
                                                        points));
                }
    //             std::cout << "Finished job " << processId << " and wrote back results\n";
                writeResultMutex.unlock();
            }
    }
}; // End Detector

#endif
