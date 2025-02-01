cmake_minimum_required(VERSION 3.10)

project(Handwriting-Recognition-Project)

set(OpenCV_DIR "C:/opencv/build")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${CMAKE_SOURCE_DIR}/src/Implementation/ImageProcessing/cpp)

set(SOURCE_FILES 
    ${CMAKE_SOURCE_DIR}/ImageProcessing/cpp/main.cpp
    ${CMAKE_SOURCE_DIR}/ImageProcessing/cpp/Binarization.cpp
    ${CMAKE_SOURCE_DIR}/ImageProcessing/cpp/LineSegmentation.cpp
    ${CMAKE_SOURCE_DIR}/ImageProcessing/cpp/Scanner.cpp
    ${CMAKE_SOURCE_DIR}/ImageProcessing/cpp/WordSegmentation.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})