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
include thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/depend.make

# Include the progress variables for this target.
include thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/progress.make

# Include the compile flags for this target's objects.
include thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/flags.make

thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o: thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/flags.make
thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o: thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int.cpp
thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o: ../thirdparty/Eigen3/doc/snippets/MatrixBase_segment_int_int.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o -c /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int.cpp

thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.i"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int.cpp > CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.i

thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.s"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int.cpp -o CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.s

# Object files for target compile_MatrixBase_segment_int_int
compile_MatrixBase_segment_int_int_OBJECTS = \
"CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o"

# External object files for target compile_MatrixBase_segment_int_int
compile_MatrixBase_segment_int_int_EXTERNAL_OBJECTS =

thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int: thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/compile_MatrixBase_segment_int_int.cpp.o
thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int: thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/build.make
thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int: thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/william/Documents/Dev/digitclassifier/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable compile_MatrixBase_segment_int_int"
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/compile_MatrixBase_segment_int_int.dir/link.txt --verbose=$(VERBOSE)
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets && ./compile_MatrixBase_segment_int_int >/home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets/MatrixBase_segment_int_int.out

# Rule to build all files generated by this target.
thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/build: thirdparty/Eigen3/doc/snippets/compile_MatrixBase_segment_int_int

.PHONY : thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/build

thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/clean:
	cd /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets && $(CMAKE_COMMAND) -P CMakeFiles/compile_MatrixBase_segment_int_int.dir/cmake_clean.cmake
.PHONY : thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/clean

thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/depend:
	cd /home/william/Documents/Dev/digitclassifier/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/william/Documents/Dev/digitclassifier /home/william/Documents/Dev/digitclassifier/thirdparty/Eigen3/doc/snippets /home/william/Documents/Dev/digitclassifier/build /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets /home/william/Documents/Dev/digitclassifier/build/thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : thirdparty/Eigen3/doc/snippets/CMakeFiles/compile_MatrixBase_segment_int_int.dir/depend

