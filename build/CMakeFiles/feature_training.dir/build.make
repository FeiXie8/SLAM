# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/xiefei/learnSLAM/learnSophus

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xiefei/learnSLAM/learnSophus/build

# Include any dependencies generated for this target.
include CMakeFiles/feature_training.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/feature_training.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/feature_training.dir/flags.make

CMakeFiles/feature_training.dir/src/feature_training.cpp.o: CMakeFiles/feature_training.dir/flags.make
CMakeFiles/feature_training.dir/src/feature_training.cpp.o: ../src/feature_training.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiefei/learnSLAM/learnSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/feature_training.dir/src/feature_training.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/feature_training.dir/src/feature_training.cpp.o -c /home/xiefei/learnSLAM/learnSophus/src/feature_training.cpp

CMakeFiles/feature_training.dir/src/feature_training.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/feature_training.dir/src/feature_training.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiefei/learnSLAM/learnSophus/src/feature_training.cpp > CMakeFiles/feature_training.dir/src/feature_training.cpp.i

CMakeFiles/feature_training.dir/src/feature_training.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/feature_training.dir/src/feature_training.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiefei/learnSLAM/learnSophus/src/feature_training.cpp -o CMakeFiles/feature_training.dir/src/feature_training.cpp.s

# Object files for target feature_training
feature_training_OBJECTS = \
"CMakeFiles/feature_training.dir/src/feature_training.cpp.o"

# External object files for target feature_training
feature_training_EXTERNAL_OBJECTS =

../bin/feature_training: CMakeFiles/feature_training.dir/src/feature_training.cpp.o
../bin/feature_training: CMakeFiles/feature_training.dir/build.make
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_img_hash.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: /usr/local/lib/libDBoW3.so
../bin/feature_training: /usr/local/lib/libopencv_world.so.4.5.2
../bin/feature_training: CMakeFiles/feature_training.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiefei/learnSLAM/learnSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/feature_training"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/feature_training.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/feature_training.dir/build: ../bin/feature_training

.PHONY : CMakeFiles/feature_training.dir/build

CMakeFiles/feature_training.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/feature_training.dir/cmake_clean.cmake
.PHONY : CMakeFiles/feature_training.dir/clean

CMakeFiles/feature_training.dir/depend:
	cd /home/xiefei/learnSLAM/learnSophus/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiefei/learnSLAM/learnSophus /home/xiefei/learnSLAM/learnSophus /home/xiefei/learnSLAM/learnSophus/build /home/xiefei/learnSLAM/learnSophus/build /home/xiefei/learnSLAM/learnSophus/build/CMakeFiles/feature_training.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/feature_training.dir/depend
