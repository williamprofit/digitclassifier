# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

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
CMAKE_SOURCE_DIR = /home/william/Documents/Dev/digitclassifier

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/william/Documents/Dev/digitclassifier/build

# Include any dependencies generated for this target.
include thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/depend.make

# Include the progress variables for this target.
include thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/flags.make

thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/sp_solver.cpp.o: thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/flags.make
thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/sp_solver.cpp.o: ../thirdparty/Eigen3/bench/spbench/sp_solver.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/sp_solver.cpp.o"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/spsolver.dir/sp_solver.cpp.o -c /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/bench/spbench/sp_solver.cpp

thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/sp_solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spsolver.dir/sp_solver.cpp.i"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/bench/spbench/sp_solver.cpp > CMakeFiles/spsolver.dir/sp_solver.cpp.i

thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/sp_solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spsolver.dir/sp_solver.cpp.s"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/bench/spbench/sp_solver.cpp -o CMakeFiles/spsolver.dir/sp_solver.cpp.s

# Object files for target spsolver
spsolver_OBJECTS = \
"CMakeFiles/spsolver.dir/sp_solver.cpp.o"

# External object files for target spsolver
spsolver_EXTERNAL_OBJECTS =

thirdparty/Eigen3/bench/spbench/spsolver: thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/sp_solver.cpp.o
thirdparty/Eigen3/bench/spbench/spsolver: thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/build.make
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libcholmod.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libcolamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libcamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libccolamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libmetis.so
thirdparty/Eigen3/bench/spbench/spsolver: thirdparty/Eigen3/blas/libeigen_blas_static.a
thirdparty/Eigen3/bench/spbench/spsolver: thirdparty/Eigen3/lapack/libeigen_lapack_static.a
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libumfpack.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libcolamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libcholmod.so
thirdparty/Eigen3/bench/spbench/spsolver: thirdparty/Eigen3/blas/libeigen_blas_static.a
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/librt.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libcamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libccolamd.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libmetis.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/libumfpack.so
thirdparty/Eigen3/bench/spbench/spsolver: /usr/lib/librt.so
thirdparty/Eigen3/bench/spbench/spsolver: thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable spsolver"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spsolver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/build: thirdparty/Eigen3/bench/spbench/spsolver

.PHONY : thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/build

thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/clean:
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench && $(CMAKE_COMMAND) -P CMakeFiles/spsolver.dir/cmake_clean.cmake
.PHONY : thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/clean

thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/depend:
	cd /home/william/Documents/Dev/digitclassifier/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/william/Documents/Dev/digitclassifier /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/bench/spbench /home/william/Documents/Dev/digitclassifier/build /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/Eigen3/bench/spbench/CMakeFiles/spsolver.dir/depend

