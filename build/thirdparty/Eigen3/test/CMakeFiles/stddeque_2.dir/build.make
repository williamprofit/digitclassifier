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
include thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/depend.make

# Include the progress variables for this target.
include thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/flags.make

thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/stddeque.cpp.o: thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/flags.make
thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/stddeque.cpp.o: ../thirdparty/Eigen3/test/stddeque.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/stddeque.cpp.o"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/stddeque_2.dir/stddeque.cpp.o -c /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/test/stddeque.cpp

thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/stddeque.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/stddeque_2.dir/stddeque.cpp.i"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/test/stddeque.cpp > CMakeFiles/stddeque_2.dir/stddeque.cpp.i

thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/stddeque.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/stddeque_2.dir/stddeque.cpp.s"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/test/stddeque.cpp -o CMakeFiles/stddeque_2.dir/stddeque.cpp.s

# Object files for target stddeque_2
stddeque_2_OBJECTS = \
"CMakeFiles/stddeque_2.dir/stddeque.cpp.o"

# External object files for target stddeque_2
stddeque_2_EXTERNAL_OBJECTS =

thirdparty/Eigen3/test/stddeque_2: thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/stddeque.cpp.o
thirdparty/Eigen3/test/stddeque_2: thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/build.make
thirdparty/Eigen3/test/stddeque_2: thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable stddeque_2"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stddeque_2.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/build: thirdparty/Eigen3/test/stddeque_2

.PHONY : thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/build

thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/clean:
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test && $(CMAKE_COMMAND) -P CMakeFiles/stddeque_2.dir/cmake_clean.cmake
.PHONY : thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/clean

thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/depend:
	cd /home/william/Documents/Dev/digitclassifier/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/william/Documents/Dev/digitclassifier /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/test /home/william/Documents/Dev/digitclassifier/build /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/Eigen3/test/CMakeFiles/stddeque_2.dir/depend

