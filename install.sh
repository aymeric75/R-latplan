#!/bin/bash

# git submodule update --recursive

# (
#     cd downward
#     ./build.py -j $(cat /proc/cpuinfo | grep -c processor) release
# )


# which ros || (
#     git clone -b release https://github.com/roswell/roswell.git
#     cd roswell
#     sh bootstrap
#     ./configure
#     make
#     make install
#     ros setup
# )

# ros delete magicffi

# ros delete sbcl

# ros dynamic-space-size=8000 install numcl arrival eazy-gnuplot 

# ros install guicho271828/magicffi

# ros install dataloader

# ros install sbcl 

# line='export PATH="/root/.roswell/bin:$PATH"'

# # Check if the line already exists in the .bashrc file
# if ! grep -Fxq "$line" ~/.bashrc
# then
#     # If the line does not exist, append it to the end of the .bashrc file
#     echo "$line" >> ~/.bashrc
#     echo "Line added to .bashrc"
# else
#     echo "Line already exists in .bashrc"
# fi

# source ~/.bashrc

make -j 1 -C lisp

./setup.py install

#./download-dataset.sh
