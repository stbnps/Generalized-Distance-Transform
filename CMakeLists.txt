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
include(cmake/OpenCVPythonModule.cmake)


# Targets ---------------------------------------------------------------------

# C++ Library
add_library(gen_dist_trans SHARED
    gen_dist_trans/src/gen_dist_trans.cpp)
target_include_directories(gen_dist_trans PUBLIC
  ${OpenCV_INCLUDE_DIRS}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gen_dist_trans/include>
  $<INSTALL_INTERFACE:include/gen_dist_trans>)
target_link_libraries(gen_dist_trans
    ${OpenCV_LIBS})
set_property(TARGET gen_dist_trans
    PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)

# Python module
if(${ENABLE_PYTHON})
    add_python_opencv_bindings(gendistrans
        PYVER "3"
        # We use directly the library header because it follows the Opencv
        # annotations conventions
        HEADERS gen_dist_trans/include/opencv2/gen_dist_trans.hpp
        # The wrapping source files would go here if there were any
        )
    target_link_libraries(gendistrans gen_dist_trans)
endif()


# Install ---------------------------------------------------------------------



# Testing ---------------------------------------------------------------------
    
if(${ENABLE_TESTING})
    enable_testing()
    
    add_executable(dontcrash
        tests/dontcrash.cpp)
    target_link_libraries(dontcrash
        ${OpenCV_LIBS}
        gen_dist_trans)
    add_test(test_dontcrash
        dontcrash)
endif()
