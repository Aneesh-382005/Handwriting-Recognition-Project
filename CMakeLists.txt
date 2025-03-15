cmake_minimum_required(VERSION 3.12)
project(Handwriting-Recognition-Project)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add pybind11 to CMake (using the submodule version)
add_subdirectory(external/pybind11)  # Assuming pybind11 is in 'external/pybind11'

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    src/Implementation/ImageProcessing/cpp/
)

# Define source and header files
set(SOURCE_FILES
    src/Implementation/ImageProcessing/cpp/Binarization.cpp
    src/Implementation/ImageProcessing/cpp/LineSegmentation.cpp
    src/Implementation/ImageProcessing/cpp/Scanner.cpp
    src/Implementation/ImageProcessing/cpp/WordSegmentation.cpp
    src/Implementation/ImageProcessing/cpp/ImageProcessing.cpp
)

set(HEADER_FILES
    src/Implementation/ImageProcessing/cpp/Binarization.hpp
    src/Implementation/ImageProcessing/cpp/LineSegmentation.hpp
    src/Implementation/ImageProcessing/cpp/Scanner.hpp
    src/Implementation/ImageProcessing/cpp/WordSegmentation.hpp
    src/Implementation/ImageProcessing/cpp/ImageProcessing.hpp
)

# Create a static library from the shared source files
add_library(SegmentationLibrary STATIC ${SOURCE_FILES} ${HEADER_FILES})

# Link OpenCV libraries to the static library
target_link_libraries(SegmentationLibrary PRIVATE ${OpenCV_LIBS})

# C++ Executable (does NOT need PyBind11)
add_executable(SegmentationApp src/Implementation/ImageProcessing/cpp/main.cpp)
target_link_libraries(SegmentationApp PRIVATE SegmentationLibrary)

# PyBind11 module (for Python)
pybind11_add_module(imgproc src/Implementation/ImageProcessing/cpp/bindings.cpp)
target_link_libraries(imgproc PRIVATE SegmentationLibrary)
