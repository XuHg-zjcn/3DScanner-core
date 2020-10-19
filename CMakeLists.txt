cmake_minimum_required(VERSION 3.16)
project(SH-3DScaner-linux)
add_definitions("-g -O3 -std=c++17 -lpthread")

#OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV found in: " ${OpenCV_INCLUDE_DIRS})

#FFTW
set(FFTW3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/fftw-3.3.8)
include_directories(${FFTW3_DIR}/api)

#project head files
include_directories(include)

#core
add_library(${PROJECT_NAME} STATIC
            src/optflow_FFT.cpp
            src/Hough.cpp
            src/Hough_core.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} fftw3 pthread)

#test program
add_executable(${PROJECT_NAME}_test
               main.cpp)
target_link_libraries(${PROJECT_NAME}_test ${OpenCV_LIBS} ${PROJECT_NAME} fftw3 pthread)
