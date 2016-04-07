cmake_minimum_required(VERSION 3.0)
project(GeneralizedDistanceTransform)


# Settings --------------------------------------------------------------------

option(ENABLE_TESTING "Compile tests")
option(ENABLE_PYTHON "Generate a python wrapper")
set(PYTHON_VERSION "" CACHE STRING "Desired python version for the bindings")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 -Wall")


# Dependencies ----------------------------------------------------------------

find_package(OpenCV REQUIRED)
include(cmake/CPPyModule.cmake)


# Targets ---------------------------------------------------------------------

# C++ Library
add_library(gendistrans SHARED
    gendistrans/src/gendistrans.cpp)
target_include_directories(gendistrans PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gendistrans/include>
    $<INSTALL_INTERFACE:include/gendistrans>)
target_link_libraries(gendistrans
    ${OpenCV_LIBS})
set_property(TARGET gendistrans
    PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Python module
if(${ENABLE_PYTHON})
    as_python_module(gendistrans_py
        NAMESPACE cv # C++ code is written as if it was an OpenCV module
        MODULE_NAME gendistrans # but we cannot reuse the cv2 module name
        # Library source files are alread formated as wrappers
        HEADERS gendistrans/include/opencv2/gendistrans.hpp
        # Add the whole source code to the module (standalone module)
        SOURCES gendistrans/src/gendistrans.cpp)
    # No linking because all source files have been put into the module
    target_include_directories(gendistrans_py PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        gendistrans/include)
endif()


# Install ---------------------------------------------------------------------



# Testing ---------------------------------------------------------------------
    
if(${ENABLE_TESTING})
    enable_testing()
    
    add_executable(dontcrash
        tests/dontcrash.cpp)
    target_link_libraries(dontcrash
        ${OpenCV_LIBS}
        gendistrans)
    add_test(test_dontcrash
        dontcrash)
endif()
