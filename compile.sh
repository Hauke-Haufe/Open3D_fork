#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#deletes build cache
rm -rf $SCRIPT_DIR/build
mkdir $SCRIPT_DIR/build

#created install dir 
mkdir -p $SCRIPT_DIR/install 

#compiles Open3d 
cd $SCRIPT_DIR/build 
cmake -DBUILD_CUDA_MODULE=true -DBUILD_LIBREALSENSE=true -DBUILD_EXAMPLES=false -DCMAKE_INSTALL_PREFIX=$SCRIPT_DIR/install -DCMAKE_BUILD_TYPE=Debug ..
make -j8
make install 
make python-package