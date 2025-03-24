cmake_minimum_required(VERSION 3.18)

# Add policy to handle CUDA module deprecation warning
cmake_policy(SET CMP0146 OLD)

project(tracker CUDA CXX)

# Specify C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Add OpenCV build directory
set(OpenCV_DIR "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/build")

# Find ZeroMQ and cppzmq
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZeroMQ REQUIRED libzmq)
find_path(CPPZMQ_INCLUDE_DIR zmq.hpp)

# Find nlohmann-json
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        json
        URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    )
    FetchContent_MakeAvailable(json)
endif()

# CUDA Configuration
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA Compiler Found: ${CMAKE_CUDA_COMPILER}")
    
    # Set CUDA architecture for RTX 30 series (Ampere)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    message(STATUS "Using CUDA architecture: ${CMAKE_CUDA_ARCHITECTURES}")
    
    # Set CUDA flags for Ampere architecture
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -arch=sm_86")
endif()

# Split OpenCV components into required and optional
set(OPENCV_REQUIRED_COMPONENTS
    core
    highgui
    imgproc
    video
    videoio
    tracking
    features2d
    calib3d
)

set(OPENCV_CUDA_COMPONENTS
    cudaimgproc
    cudawarping
    cudafeatures2d
    cudafilters
)

# Find required OpenCV components first
find_package(OpenCV 4.10.0 REQUIRED COMPONENTS ${OPENCV_REQUIRED_COMPONENTS})

# Try to find CUDA components, but don't require them
foreach(component ${OPENCV_CUDA_COMPONENTS})
    find_package(OpenCV 4.10.0 QUIET OPTIONAL_COMPONENTS ${component})
endforeach()

# Find CUDA without version restriction
if(EXISTS "/usr/local/cuda-12.6")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
elseif(EXISTS "/usr/local/cuda-12.0")
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.0")
else()
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
endif()

find_package(CUDA REQUIRED)
message(STATUS "Found CUDA ${CUDA_VERSION} at ${CUDA_TOOLKIT_ROOT_DIR}")

# Print configuration information
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "ZeroMQ Include Dirs: ${ZeroMQ_INCLUDE_DIRS}")
message(STATUS "ZeroMQ Libraries: ${ZeroMQ_LIBRARIES}")

# Add executable
add_executable(tracker optimized_tracker.cpp)

# Include directories
target_include_directories(tracker PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${ZeroMQ_INCLUDE_DIRS}
    ${CPPZMQ_INCLUDE_DIR}
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/core/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/calib3d/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/features2d/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/highgui/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/imgcodecs/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/imgproc/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/video/include"
    "$ENV{HOME}/Downloads/opencv/opencv-4.10.0/modules/videoio/include"
    "$ENV{HOME}/Downloads/opencv/opencv_contrib-4.10.0/modules/tracking/include"
)

# Add CUDA module includes only if available
if(OpenCV_CUDA_COMPONENTS_FOUND)
    target_include_directories(tracker PRIVATE
        "$ENV{HOME}/Downloads/opencv/opencv_contrib-4.10.0/modules/cudaimgproc/include"
        "$ENV{HOME}/Downloads/opencv/opencv_contrib-4.10.0/modules/cudawarping/include"
        "$ENV{HOME}/Downloads/opencv/opencv_contrib-4.10.0/modules/cudafeatures2d/include"
        "$ENV{HOME}/Downloads/opencv/opencv_contrib-4.10.0/modules/cudafilters/include"
        "$ENV{HOME}/Downloads/opencv/opencv_contrib-4.10.0/modules/cudacodec/include"
    )
endif()

# Link libraries
target_link_libraries(tracker PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_CUDART_LIBRARY}
    ${ZeroMQ_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Set CUDA-specific properties
if(CMAKE_CUDA_COMPILER)
    set_target_properties(tracker PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(tracker PRIVATE WITH_CUDA)
endif()

# Set build type to Release for better performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Print final configuration
message(STATUS "")
message(STATUS "Build Configuration Summary:")
message(STATUS "============================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Toolkit Path: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "ZeroMQ Found: ${ZeroMQ_FOUND}")
message(STATUS "")
message(STATUS "Available OpenCV CUDA modules:")
foreach(component ${OPENCV_CUDA_COMPONENTS})
    if(OpenCV_${component}_FOUND)
        message(STATUS "  - ${component}: FOUND")
    else()
        message(STATUS "  - ${component}: NOT FOUND")
    endif()
endforeach()
message(STATUS "============================")
