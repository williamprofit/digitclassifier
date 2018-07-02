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
include thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/depend.make

# Include the progress variables for this target.
include thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/flags.make

thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/make_circulant.cpp.o: thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/flags.make
thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/make_circulant.cpp.o: ../thirdparty/Eigen3/doc/examples/make_circulant.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/make_circulant.cpp.o"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/make_circulant.dir/make_circulant.cpp.o -c /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/doc/examples/make_circulant.cpp

thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/make_circulant.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/make_circulant.dir/make_circulant.cpp.i"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/doc/examples/make_circulant.cpp > CMakeFiles/make_circulant.dir/make_circulant.cpp.i

thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/make_circulant.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/make_circulant.dir/make_circulant.cpp.s"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/doc/examples/make_circulant.cpp -o CMakeFiles/make_circulant.dir/make_circulant.cpp.s

# Object files for target make_circulant
make_circulant_OBJECTS = \
"CMakeFiles/make_circulant.dir/make_circulant.cpp.o"

# External object files for target make_circulant
make_circulant_EXTERNAL_OBJECTS =

thirdparty/Eigen3/doc/examples/make_circulant: thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/make_circulant.cpp.o
thirdparty/Eigen3/doc/examples/make_circulant: thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/build.make
thirdparty/Eigen3/doc/examples/make_circulant: thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable make_circulant"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/make_circulant.dir/link.txt --verbose=$(VERBOSE)
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples && ./make_circulant >/home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples/make_circulant.out

# Rule to build all files generated by this target.
thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/build: thirdparty/Eigen3/doc/examples/make_circulant

.PHONY : thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/build

thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/clean:
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples && $(CMAKE_COMMAND) -P CMakeFiles/make_circulant.dir/cmake_clean.cmake
.PHONY : thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/clean

thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/depend:
	cd /home/william/Documents/Dev/digitclassifier/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/william/Documents/Dev/digitclassifier /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/doc/examples /home/william/Documents/Dev/digitclassifier/build /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/Eigen3/doc/examples/CMakeFiles/make_circulant.dir/depend

