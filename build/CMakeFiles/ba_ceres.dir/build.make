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
include CMakeFiles/ba_ceres.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/ba_ceres.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ba_ceres.dir/flags.make

CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.o: CMakeFiles/ba_ceres.dir/flags.make
CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.o: ../src/ba_ceres.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiefei/learnSLAM/learnSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.o -c /home/xiefei/learnSLAM/learnSophus/src/ba_ceres.cpp

CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiefei/learnSLAM/learnSophus/src/ba_ceres.cpp > CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.i

CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiefei/learnSLAM/learnSophus/src/ba_ceres.cpp -o CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.s

CMakeFiles/ba_ceres.dir/src/common.cpp.o: CMakeFiles/ba_ceres.dir/flags.make
CMakeFiles/ba_ceres.dir/src/common.cpp.o: ../src/common.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/xiefei/learnSLAM/learnSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ba_ceres.dir/src/common.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/ba_ceres.dir/src/common.cpp.o -c /home/xiefei/learnSLAM/learnSophus/src/common.cpp

CMakeFiles/ba_ceres.dir/src/common.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ba_ceres.dir/src/common.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/xiefei/learnSLAM/learnSophus/src/common.cpp > CMakeFiles/ba_ceres.dir/src/common.cpp.i

CMakeFiles/ba_ceres.dir/src/common.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ba_ceres.dir/src/common.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/xiefei/learnSLAM/learnSophus/src/common.cpp -o CMakeFiles/ba_ceres.dir/src/common.cpp.s

# Object files for target ba_ceres
ba_ceres_OBJECTS = \
"CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.o" \
"CMakeFiles/ba_ceres.dir/src/common.cpp.o"

# External object files for target ba_ceres
ba_ceres_EXTERNAL_OBJECTS =

../bin/ba_ceres: CMakeFiles/ba_ceres.dir/src/ba_ceres.cpp.o
../bin/ba_ceres: CMakeFiles/ba_ceres.dir/src/common.cpp.o
../bin/ba_ceres: CMakeFiles/ba_ceres.dir/build.make
../bin/ba_ceres: /usr/local/lib/libceres.a
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libglog.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libgflags.so.2.2.2
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libspqr.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libtbb.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libcholmod.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libccolamd.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libcamd.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libcolamd.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libamd.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/liblapack.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libblas.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/librt.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/liblapack.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libblas.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libsuitesparseconfig.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/librt.so
../bin/ba_ceres: /usr/lib/x86_64-linux-gnu/libcxsparse.so
../bin/ba_ceres: CMakeFiles/ba_ceres.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/xiefei/learnSLAM/learnSophus/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/ba_ceres"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ba_ceres.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ba_ceres.dir/build: ../bin/ba_ceres

.PHONY : CMakeFiles/ba_ceres.dir/build

CMakeFiles/ba_ceres.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ba_ceres.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ba_ceres.dir/clean

CMakeFiles/ba_ceres.dir/depend:
	cd /home/xiefei/learnSLAM/learnSophus/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/xiefei/learnSLAM/learnSophus /home/xiefei/learnSLAM/learnSophus /home/xiefei/learnSLAM/learnSophus/build /home/xiefei/learnSLAM/learnSophus/build /home/xiefei/learnSLAM/learnSophus/build/CMakeFiles/ba_ceres.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ba_ceres.dir/depend

