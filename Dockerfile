FROM ubuntu:18.04

# Disable interaction with tzinf, which asks for your geographic region
ENV DEBIAN_FRONTEND=noninteractive

# update repos and get packages
RUN apt-get update &&                          \
    apt-get install -y \
        sudo git ssh wget cmake python3-pip python3-numpy cmake libgoogle-glog-dev libatlas-base-dev libopencv-dev \
         libboost-all-dev libeigen3-dev libsuitesparse-dev libgtk2.0-dev libsm6 libxext6 wget unzip mesa-utils texlive-latex-base
RUN pip3 install --upgrade pip
RUN pip3 install wheel Pillow scipy opencv-python matplotlib Cython pyx commentjson tqdm

## Container's mount point for the host's input/output folder
VOLUME "/host"

## Enable X in the container
ARG DISPLAY
ENV XAUTHORITY $XAUTHORITY

## Setup "machine id" used by DBus for proper (complaint-free) X usage
ARG machine_id
ENV machine_id=${machine_id}
RUN sudo chmod o+w /etc/machine-id &&       \
    echo ${machine_id} > /etc/machine-id && \
sudo chmod o-w /etc/machine-id

## Switch to non-root user
ARG uid
ARG gid
ARG username
ENV uid=${uid}
ENV gid=${gid}
ENV USER=${username}
RUN groupadd -g $gid $USER &&                                         \
    mkdir -p /home/$USER &&                                           \
    echo "${USER}:x:${uid}:${gid}:${USER},,,:/home/${USER}:/bin/bash" \
         >> /etc/passwd &&                                            \
    echo "${USER}:x:${uid}:"                                          \
         >> /etc/group &&                                             \
    echo "${USER} ALL=(ALL) NOPASSWD: ALL"                            \
         > /etc/sudoers.d/${USER} &&                                  \
    chmod 0440 /etc/sudoers.d/${USER} &&                              \
    chown ${uid}:${gid} -R /home/${USER}

USER ${USER}
ENV HOME=/home/${USER}

WORKDIR ${HOME}

## build and install ceres
RUN cd ~ && wget http://ceres-solver.org/ceres-solver-1.14.0.tar.gz && tar -zxf ceres-solver-1.14.0.tar.gz && mkdir ceres-solver-1.14.0/build  && cd ceres-solver-1.14.0/build && cmake .. && make -j6 && sudo make install && sudo make install

## make python3 default
RUN sudo rm -f /usr/bin/python && sudo ln -s /usr/bin/python3 /usr/bin/python

## installation of FreiCalib
RUN cd ~ && wget --no-check-certificate https://lmb.informatik.uni-freiburg.de/data/RatTrack/data/FreiCalib-master.zip && unzip FreiCalib-master.zip && rm FreiCalib-master.zip && cd FreiCalib-master/
RUN cd ~/FreiCalib-master/TagDetector && python setupBatch.py build_ext --inplace
RUN cd ~/FreiCalib-master/Bundle && mkdir build && cd build && cmake .. && make
