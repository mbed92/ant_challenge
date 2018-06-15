cmake_minimum_required(VERSION 3.8)
project(TooploxBlurDepth)

# clion cannot see CUDA otherwise
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-8.0")

FIND_PACKAGE(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_FILES main.cpp
        include/utilities.h src/utilities.cpp
        include/IEstimator.h src/IEstimator.cpp
        include/BlurEstimator.h src/BlurEstimator.cpp)
set(DEPENDENCIES ${OpenCV_LIBS})

add_executable(TooploxBlurDepth ${SOURCE_FILES})
target_link_libraries(TooploxBlurDepth ${DEPENDENCIES})