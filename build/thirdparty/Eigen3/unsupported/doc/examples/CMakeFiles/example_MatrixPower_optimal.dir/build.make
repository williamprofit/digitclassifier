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
include thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/depend.make

# Include the progress variables for this target.
include thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/flags.make

thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.o: thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/flags.make
thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.o: ../thirdparty/Eigen3/unsupported/doc/examples/MatrixPower_optimal.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.o"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.o -c /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/unsupported/doc/examples/MatrixPower_optimal.cpp

thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.i"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/unsupported/doc/examples/MatrixPower_optimal.cpp > CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.i

thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.s"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/unsupported/doc/examples/MatrixPower_optimal.cpp -o CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.s

# Object files for target example_MatrixPower_optimal
example_MatrixPower_optimal_OBJECTS = \
"CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.o"

# External object files for target example_MatrixPower_optimal
example_MatrixPower_optimal_EXTERNAL_OBJECTS =

thirdparty/Eigen3/unsupported/doc/examples/example_MatrixPower_optimal: thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/MatrixPower_optimal.cpp.o
thirdparty/Eigen3/unsupported/doc/examples/example_MatrixPower_optimal: thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/build.make
thirdparty/Eigen3/unsupported/doc/examples/example_MatrixPower_optimal: thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example_MatrixPower_optimal"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example_MatrixPower_optimal.dir/link.txt --verbose=$(VERBOSE)
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples && ./example_MatrixPower_optimal >/home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples/MatrixPower_optimal.out

# Rule to build all files generated by this target.
thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/build: thirdparty/Eigen3/unsupported/doc/examples/example_MatrixPower_optimal

.PHONY : thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/build

thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/clean:
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/example_MatrixPower_optimal.dir/cmake_clean.cmake
.PHONY : thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/clean

thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/depend:
	cd /home/william/Documents/Dev/digitclassifier/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/william/Documents/Dev/digitclassifier /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/unsupported/doc/examples /home/william/Documents/Dev/digitclassifier/build /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/Eigen3/unsupported/doc/examples/CMakeFiles/example_MatrixPower_optimal.dir/depend

