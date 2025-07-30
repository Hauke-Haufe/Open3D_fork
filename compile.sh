#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#deletes build cache
sudo rm -rf $SCRIPT_DIR/build
mkdir $SCRIPT_DIR/build

#created install dir 
sudo mkdir -p /usr/local/lib/Open3D_fork

#compiles Open3d 
cd $SCRIPT_DIR/build 
cmake -DBUILD_CUDA_MODULE=true -DPYTHON_EXECUTABLE=$(which python)-DBUILD_LIBREALSENSE=true -DBUILD_EXAMPLES=false -DCMAKE_INSTALL_PREFIX=/usr/local/lib/Open3D_fork -DCMAKE_BUILD_TYPE=Debug ..
make -j6
sudo make install 
sudo make install-pip-package