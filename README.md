# FreiCalib

This is the calibration software developed in conjunction with our recent publication

    @TechReport{Freipose2020,
        author    = {Christian Zimmermann, Artur Schneider, Mansour Alyahyay, Thomas Brox and Ilka Diester},
        title     = {FreiPose: A Deep Learning Framework for Precise Animal Motion Capture in 3D Spaces},
        year      = {2020},
        url          = {"https://lmb.informatik.uni-freiburg.de/projects/freipose/"}
    }

It allows to calibrate a multi camera rig from dedicated sequences using a calibration object. 
Help to create such a calibration object is provided

    
## Tutorial using Docker

We provide a small example on how this software can be used and recommend to run it inside the Docker image. 
If you are not familiar with Docker we highly recommend checking out https://docs.docker.com/ to get an idea on what it does.
In an nutshell it provides an environment for software to run in that is mostly isolated from the host machine, 
which gives great control over dependencies and libraries. On the other hand passing data between the host machine and 
the Docker image has to be dealt with and also showing Windows (X-forwarding) can cause problems.
 

Install docker on your system according to https://docs.docker.com/install/

Clone OR download repository:

    git clone https://github.com/zimmerm/FreiCalib
    wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/RatTrack/data/FreiCalib-master.zip && unzip FreiCalib-master.zip && mv FreiCalib-master FreiCalib
    

Define path to the FreiCalib folder:

    FC_DIR="${PWD}/FreiCalib/"
    
Download example data:
    
     cd ${FC_DIR}/data && wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/RatTrack/data/calib.zip && unzip calib.zip && rm calib.zip

Build docker container (probably will take ~10 minutes):
    
    cd ${FC_DIR} && make
    
Start the docker container:
    
    bash docker_run.sh
    
If you peek inside docker_run.sh you can see that we mount the folder `${FC_DIR}/data` to `/host/` inside the container.
This is the point of data exchange between your host machine and the docker container.

Calculate calibration (inside the container):

    cd ~/FreiCalib-master/ && python calib_M.py /host/marker_32h11b2_4x10x_5cm.json /host/run000_cam1.avi
    
Reprojection error before bundling: 7.79 pixel  *

Final reprojection error: 1.10 pixel *

Notes:
    
    - When a window pops up it can be closed by pressing an arbitraty key 
    - Known issue: If the window is closed by a mouse click on the respective window closing button the programm will not end. Killing it with CTRL+C is necessary to resume.
    - If you want to use videos: Simply point it to one of the video files. The program will try to match cam0, cam1, etc to find the other files
    - Instead of giving it a video file you can also provide it with a folder, where subfolders called cam0, cam1, cam2 branch off.
    - To provide data you can put the data into ${FC_DIR}/data on the host machine (or create symlinks there) or alternatively change the path that is mounted to /host/ in the container
    - If you change the folder mounted: All directories up to the folder mounted to `/host/` need o+x rights (default: `${FC_DIR}/data`)


Check calibration on same sequence (inside the container):

    cd ~/FreiCalib-master/ && python check_M.py /host/marker_32h11b2_4x10x_5cm.json /host/run000_cam1.avi /host/M.json
    
Final reprojection error: 1.84 pixel (Larger than before due to non optimized object poses)

Check calibration on another sequence (inside the container):

    cd ~/FreiCalib-master/ && python check_M.py /host/marker_32h11b2_4x10x_5cm.json /host/run001_cam1.avi /host/M.json

Final reprojection error: 2.40 pixel *


*: exact value will differ due to some randomness
   

## User guide

### Create a new calibration pattern

When creating a new pattern you will need to physically create an object that you move in front of your cameras
while recording a calibration video sequence.  

Example call:

    python create_marker.py --nx 10 --ny 4 --tsize 0.05 --double
    
Which will give you three files in `./tags/`. Two PDF's which are the front and back side of the calibration object and 
a json file containing information to be passed to subsequent scripts in order to work with this specific pattern.
This call will create a double sided (front and back) marker where each cell is 5 cm large and where patterns are arranged
in a 4 by 10 grid. In total there are 2x4x10=80 patterns.
    
### Calibration: Intrinsic

Intrinsic calibration will determine imaging parameters of a single cameras. If you wish to calculate this explicitly you
 can also use this routine in isolation.
 
Example call:

    python calib_K.py $MARKER_PATH $DATA_PATH

Where `$MARKER_PATH` should point to the json file created by create_marker.py and `$DATA_PATH` either points to a directory
 of images or a video file. Supported file types are: 'jpg', 'jpeg', 'png' and 'bmp'. All video files supported by OpenCV can be used.
 
If you wish to check an intrinsic calibration and get some idea on how to improve the calibration process two visualization scripts are being provided:

    python vis_det.py $MARKER_PATH $DATA_PATH
    python vis_K.py $MARKER_PATH $DATA_PATH
    
Certain names relative to the image/video data are being assumed while calling these scripts. 
When dealing with intrinsic calibration the assumed names are:

    - Calibration board detections are being stored in detection.json files
    - Intrinsic calibration is stored as K.json
    
But these are changeable via additional arguments while calling the scripts. 
All scripts provide help on their usage when being called with the `--help` argument.   
 
### Calibration: Extrinsic
 
Extrinsic calibration will determine the relative position and orientation of the cameras with respect to each other.
 
Example call:

    python calib_M.py $MARKER_PATH $DATA_PATH
    
This script will dump:

    - Two files per camera
    - For each camera one detections_cam%d.json and one K_cam%d.json
    - One M.json containing the extrinsic calibration
    
To check the calibration the following script can be used: 

    python check_M.py $MARKER_PATH $DATA_PATH $CALIB_PATH

Where `$MARKER_PATH` should point to the json file created by create_marker.py and `$DATA_PATH` either points to a directory
 of images or a video file. Supported file types are: 'jpg', 'jpeg', 'png' and 'bmp'. All video files supported by OpenCV can be used. `$CALIB_PATH` is the M.json to be used.


