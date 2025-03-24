# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/skand/opencv_cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/skand/opencv_cpp/build

# Include any dependencies generated for this target.
include CMakeFiles/tracker.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/tracker.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/tracker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/tracker.dir/flags.make

CMakeFiles/tracker.dir/optimized_tracker.cpp.o: CMakeFiles/tracker.dir/flags.make
CMakeFiles/tracker.dir/optimized_tracker.cpp.o: /home/skand/opencv_cpp/optimized_tracker.cpp
CMakeFiles/tracker.dir/optimized_tracker.cpp.o: CMakeFiles/tracker.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/skand/opencv_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/tracker.dir/optimized_tracker.cpp.o"
	/usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/tracker.dir/optimized_tracker.cpp.o -MF CMakeFiles/tracker.dir/optimized_tracker.cpp.o.d -o CMakeFiles/tracker.dir/optimized_tracker.cpp.o -c /home/skand/opencv_cpp/optimized_tracker.cpp

CMakeFiles/tracker.dir/optimized_tracker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/tracker.dir/optimized_tracker.cpp.i"
	/usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/skand/opencv_cpp/optimized_tracker.cpp > CMakeFiles/tracker.dir/optimized_tracker.cpp.i

CMakeFiles/tracker.dir/optimized_tracker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/tracker.dir/optimized_tracker.cpp.s"
	/usr/bin/g++-10 $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/skand/opencv_cpp/optimized_tracker.cpp -o CMakeFiles/tracker.dir/optimized_tracker.cpp.s

# Object files for target tracker
tracker_OBJECTS = \
"CMakeFiles/tracker.dir/optimized_tracker.cpp.o"

# External object files for target tracker
tracker_EXTERNAL_OBJECTS =

tracker: CMakeFiles/tracker.dir/optimized_tracker.cpp.o
tracker: CMakeFiles/tracker.dir/build.make
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_tracking.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_cudaimgproc.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_cudafeatures2d.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_cudafilters.so.4.10.0
tracker: /usr/local/cuda-12.6/lib64/libcudart_static.a
tracker: /usr/lib/x86_64-linux-gnu/librt.a
tracker: /usr/lib/x86_64-linux-gnu/libcuda.so
tracker: /usr/local/cuda-12.6/lib64/libcudart.so
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_highgui.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_video.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_videoio.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_calib3d.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_plot.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_datasets.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_imgcodecs.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_text.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_dnn.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_ml.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_features2d.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_cudawarping.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_imgproc.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_flann.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_cudaarithm.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_core.so.4.10.0
tracker: /home/skand/Downloads/opencv/opencv-4.10.0/build/lib/libopencv_cudev.so.4.10.0
tracker: CMakeFiles/tracker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/skand/opencv_cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable tracker"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/tracker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/tracker.dir/build: tracker
.PHONY : CMakeFiles/tracker.dir/build

CMakeFiles/tracker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/tracker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/tracker.dir/clean

CMakeFiles/tracker.dir/depend:
	cd /home/skand/opencv_cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/skand/opencv_cpp /home/skand/opencv_cpp /home/skand/opencv_cpp/build /home/skand/opencv_cpp/build /home/skand/opencv_cpp/build/CMakeFiles/tracker.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/tracker.dir/depend

