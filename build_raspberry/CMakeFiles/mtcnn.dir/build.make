# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pi/TuanDX/face-detection-MTCNN-ncnn

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry

# Include any dependencies generated for this target.
include CMakeFiles/mtcnn.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mtcnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mtcnn.dir/flags.make

CMakeFiles/mtcnn.dir/main.o: CMakeFiles/mtcnn.dir/flags.make
CMakeFiles/mtcnn.dir/main.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mtcnn.dir/main.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mtcnn.dir/main.o -c /home/pi/TuanDX/face-detection-MTCNN-ncnn/main.cpp

CMakeFiles/mtcnn.dir/main.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mtcnn.dir/main.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/TuanDX/face-detection-MTCNN-ncnn/main.cpp > CMakeFiles/mtcnn.dir/main.i

CMakeFiles/mtcnn.dir/main.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mtcnn.dir/main.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/TuanDX/face-detection-MTCNN-ncnn/main.cpp -o CMakeFiles/mtcnn.dir/main.s

CMakeFiles/mtcnn.dir/Mtcnn.o: CMakeFiles/mtcnn.dir/flags.make
CMakeFiles/mtcnn.dir/Mtcnn.o: ../Mtcnn.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mtcnn.dir/Mtcnn.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mtcnn.dir/Mtcnn.o -c /home/pi/TuanDX/face-detection-MTCNN-ncnn/Mtcnn.cpp

CMakeFiles/mtcnn.dir/Mtcnn.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mtcnn.dir/Mtcnn.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/TuanDX/face-detection-MTCNN-ncnn/Mtcnn.cpp > CMakeFiles/mtcnn.dir/Mtcnn.i

CMakeFiles/mtcnn.dir/Mtcnn.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mtcnn.dir/Mtcnn.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/TuanDX/face-detection-MTCNN-ncnn/Mtcnn.cpp -o CMakeFiles/mtcnn.dir/Mtcnn.s

CMakeFiles/mtcnn.dir/KalmanTracker.o: CMakeFiles/mtcnn.dir/flags.make
CMakeFiles/mtcnn.dir/KalmanTracker.o: ../KalmanTracker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/mtcnn.dir/KalmanTracker.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mtcnn.dir/KalmanTracker.o -c /home/pi/TuanDX/face-detection-MTCNN-ncnn/KalmanTracker.cc

CMakeFiles/mtcnn.dir/KalmanTracker.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mtcnn.dir/KalmanTracker.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/TuanDX/face-detection-MTCNN-ncnn/KalmanTracker.cc > CMakeFiles/mtcnn.dir/KalmanTracker.i

CMakeFiles/mtcnn.dir/KalmanTracker.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mtcnn.dir/KalmanTracker.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/TuanDX/face-detection-MTCNN-ncnn/KalmanTracker.cc -o CMakeFiles/mtcnn.dir/KalmanTracker.s

CMakeFiles/mtcnn.dir/Hungarian.o: CMakeFiles/mtcnn.dir/flags.make
CMakeFiles/mtcnn.dir/Hungarian.o: ../Hungarian.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/mtcnn.dir/Hungarian.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mtcnn.dir/Hungarian.o -c /home/pi/TuanDX/face-detection-MTCNN-ncnn/Hungarian.cc

CMakeFiles/mtcnn.dir/Hungarian.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mtcnn.dir/Hungarian.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pi/TuanDX/face-detection-MTCNN-ncnn/Hungarian.cc > CMakeFiles/mtcnn.dir/Hungarian.i

CMakeFiles/mtcnn.dir/Hungarian.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mtcnn.dir/Hungarian.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pi/TuanDX/face-detection-MTCNN-ncnn/Hungarian.cc -o CMakeFiles/mtcnn.dir/Hungarian.s

# Object files for target mtcnn
mtcnn_OBJECTS = \
"CMakeFiles/mtcnn.dir/main.o" \
"CMakeFiles/mtcnn.dir/Mtcnn.o" \
"CMakeFiles/mtcnn.dir/KalmanTracker.o" \
"CMakeFiles/mtcnn.dir/Hungarian.o"

# External object files for target mtcnn
mtcnn_EXTERNAL_OBJECTS =

mtcnn: CMakeFiles/mtcnn.dir/main.o
mtcnn: CMakeFiles/mtcnn.dir/Mtcnn.o
mtcnn: CMakeFiles/mtcnn.dir/KalmanTracker.o
mtcnn: CMakeFiles/mtcnn.dir/Hungarian.o
mtcnn: CMakeFiles/mtcnn.dir/build.make
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_shape.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_stitching.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_superres.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_videostab.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_aruco.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_bgsegm.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_bioinspired.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_ccalib.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_datasets.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_dpm.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_face.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_freetype.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_fuzzy.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_hdf.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_line_descriptor.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_optflow.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_plot.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_reg.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_saliency.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_stereo.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_structured_light.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_surface_matching.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_text.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_ximgproc.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_xobjdetect.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_xphoto.so.3.2.0
mtcnn: ../packages/ncnn_20171225/lib/linux/armv7l/gcc6.3/libncnn.a
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_video.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_viz.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_phase_unwrapping.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_rgbd.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_calib3d.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_features2d.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_flann.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_objdetect.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_ml.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_highgui.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_photo.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_videoio.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_imgcodecs.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_imgproc.so.3.2.0
mtcnn: /usr/lib/arm-linux-gnueabihf/libopencv_core.so.3.2.0
mtcnn: CMakeFiles/mtcnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX executable mtcnn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mtcnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mtcnn.dir/build: mtcnn

.PHONY : CMakeFiles/mtcnn.dir/build

CMakeFiles/mtcnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mtcnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mtcnn.dir/clean

CMakeFiles/mtcnn.dir/depend:
	cd /home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pi/TuanDX/face-detection-MTCNN-ncnn /home/pi/TuanDX/face-detection-MTCNN-ncnn /home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry /home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry /home/pi/TuanDX/face-detection-MTCNN-ncnn/build_raspberry/CMakeFiles/mtcnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mtcnn.dir/depend

