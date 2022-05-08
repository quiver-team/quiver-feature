cmake_minimum_required(VERSION 3.12)
project(quiver_feature)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

file(GLOB HEADERS csrc/include/qvf/*.h csrc/include/infinity/*.h csrc/include/infinity/core/*.h csrc/include/infinity/memory/*.h csrc/include/infinity/queues/*.h csrc/include/infinity/requests/*.h csrc/include/infinity/utils/*.h)
file(GLOB SOURCES csrc/src/*.cpp csrc/include/infinity/requests/*.cpp csrc/include/infinity/core/*.cpp csrc/include/infinity/memory/*.cpp csrc/include/infinity/queues/*.cpp csrc/include/infinity/utils/*.cpp)
file(GLOB TEST_SOURCES tests/cpp/*.cpp)

set_source_files_properties(SOURCES PROPERTIES COMPILE_OPTIONS "-libverbs")
set_source_files_properties(TEST_SOURCES PROPERTIES COMPILE_OPTIONS "-libverbs")

find_package(Python3 COMPONENTS Interpreter Development)
find_package(Torch REQUIRED)
add_library(${PROJECT_NAME} SHARED ${SOURCES})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)
target_link_libraries(${PROJECT_NAME} PRIVATE ibverbs)

if (PROF)
  target_link_options(${PROJECT_NAME} PRIVATE "-pg")
endif()

target_include_directories(${PROJECT_NAME} PUBLIC csrc/include)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

if(BUILD_TEST)
  add_executable(cpp_test ${TEST_SOURCES})
  target_link_libraries(cpp_test PRIVATE ${TORCH_LIBRARIES})
  target_link_libraries(cpp_test PRIVATE Python3::Python)
  target_link_libraries(cpp_test PRIVATE ${PROJECT_NAME})
endif()
