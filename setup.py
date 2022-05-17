import os
import sys
import glob
import os.path as osp
from itertools import product
from setuptools import setup, find_packages
import platform

import torch
from torch.__config__ import parallel_info
from torch.utils.cpp_extension import BuildExtension
from torch.utils.cpp_extension import CppExtension, CUDAExtension, CUDA_HOME
import torch.utils.cpp_extension as cpp_extension

WITH_CUDA = torch.cuda.is_available() and CUDA_HOME is not None
suffices = ['cpu', 'cuda'] if WITH_CUDA else ['cpu']
if os.getenv('FORCE_CUDA', '0') == '1':
    suffices = ['cuda', 'cpu']
if os.getenv('FORCE_ONLY_CUDA', '0') == '1':
    suffices = ['cuda']
if os.getenv('FORCE_ONLY_CPU', '0') == '1':
    suffices = ['cpu']

BUILD_DOCS = os.getenv('BUILD_DOCS', '0') == '1'

WITH_SYMBOLS = True if os.getenv('WITH_SYMBOLS', '0') == '1' else False


def get_torch_includes():
    lib_include = os.path.join(cpp_extension._TORCH_PATH, 'include')
    paths = [
        osp.join(lib_include, 'ATen'),
        osp.join(lib_include, 'c10'),
        osp.join(lib_include, 'caffe2'),
    ]

    return paths


def get_extensions():
    extensions = []
    libraries = ['ibverbs']

    extensions_dir = osp.join('csrc')

    srcs = glob.glob(osp.join(extensions_dir, 'src', '*.cpp'))
    srcs += glob.glob(osp.join(extensions_dir, 'src', '*.cu'))
    srcs += glob.glob(osp.join(extensions_dir, 'include', "infinity/core", '*.cpp'))
    srcs += glob.glob(osp.join(extensions_dir, 'include', "infinity/memory", '*.cpp'))
    srcs += glob.glob(osp.join(extensions_dir, 'include', "infinity/queues", '*.cpp'))
    srcs += glob.glob(osp.join(extensions_dir, 'include', "infinity/requests", '*.cpp'))
    srcs += glob.glob(osp.join(extensions_dir, 'include', "infinity/utils", '*.cpp'))
    srcs += glob.glob(osp.join(extensions_dir, 'include', "miniz", '*.c'))
    includes = osp.join(extensions_dir, 'include/')

    define_macros = [('WITH_PYTHON', None)]
    extra_compile_args = {
        'cxx': ['-O3', '-std=c++17', '-libverbs'],
        '/usr/local/cuda/bin/nvcc': ['-O3', '--expt-extended-lambda', '-std=c++17', '-libverbs']}
    extra_link_args = [] if WITH_SYMBOLS else ['-s']

    Extension = CUDAExtension
    extension = Extension(
        'qvf',
        srcs,
        include_dirs=[includes] + get_torch_includes(),
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        libraries=libraries,
    )
    extensions += [extension]
    print()
    return extensions


install_requires = []
setup_requires = []
tests_require = ['pytest', 'pytest-runner', 'pytest-cov']

setup(
    name='quiver_feature',
    version='0.0.1',
    author='quiver-team',
    author_email='',
    url='https://github.com/quiver-team/quiver_feature',
    description=('PyTorch Library for graph learning sampling'),
    keywords=['pytorch', 'sparse', 'graph'],
    license='MIT',
    python_requires='>=3.6',
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require={'test': tests_require},
    ext_modules=get_extensions() if not BUILD_DOCS else [],
    cmdclass={
        'build_ext':
            BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)
    },
    packages=find_packages(),
)
