rm -rf build
mkdir -p build
cd build
Torch_DIR=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` \
cmake -DCMAKE_INSTALL_PREFIX=. ..
make install