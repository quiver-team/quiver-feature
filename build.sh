# build infinity
make all
rm -rf build
mkdir -p build
cd build
Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
cmake -DBUILD_TEST=0 -DPROF=0 -DCMAKE_INSTALL_PREFIX=. ..
make install