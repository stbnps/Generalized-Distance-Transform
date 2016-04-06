set(this_file_directory ${CMAKE_CURRENT_LIST_DIR})

# Dependencies ----------------------------------------------------------------

include(CMakeParseArguments)


function(add_python_opencv_bindings MODULE_NAME)

    # Parse arguments ---------------------------------------------------------
    set(options)
    set(oneValueArgs PYVER)
    set(multiValueArgs HEADERS)
    cmake_parse_arguments(ARG
        "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Load dependencies  ------------------------------------------------------

    set(Python_ADDITIONAL_VERSIONS ${ARG_PYVER})
    find_package(OpenCV REQUIRED)
    find_package(PythonInterp REQUIRED)
    find_package(PythonLibs REQUIRED)
    if(NOT ("${ARG_PYVER}" STREQUAL "") AND NOT ("${PYTHON_VERSION_MAJOR}" EQUAL "${ARG_PYVER}"))
        message(FATAL_ERROR "Python version ${PYVER} was not found (note: \
            found ${PYTHON_VERSION_STRING}).")
    endif()

    # Source code generation --------------------------------------------------

    # List header files in headers.txt
    set(gen_files_path ${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME})
    file(MAKE_DIRECTORY ${gen_files_path})
    message("headers: ${ARG_HEADERS}")
    file(WRITE ${gen_files_path}/headers.txt "${ARG_HEADERS}")

    # Parse headers and generate python bindings
    add_custom_command(
        OUTPUT
        ${gen_files_path}/pyopencv_generated_include.h
        ${gen_files_path}/pyopencv_generated_funcs.h
        ${gen_files_path}/pyopencv_generated_ns_reg.h
        ${gen_files_path}/pyopencv_generated_type_reg.h
        ${gen_files_path}/pyopencv_generated_types.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${PYTHON_EXECUTABLE} ${this_file_directory}/gen2.py
            ${gen_files_path}
            ${gen_files_path}/headers.txt)
    
    # Make a target for cmake
    add_custom_target(generated_headers
        DEPENDS
        ${gen_files_path}/pyopencv_generated_include.h
        ${gen_files_path}/pyopencv_generated_funcs.h
        ${gen_files_path}/pyopencv_generated_ns_reg.h
        ${gen_files_path}/pyopencv_generated_type_reg.h
        ${gen_files_path}/pyopencv_generated_types.h)

    # Generate configuration file
    configure_file(${this_file_directory}/module_config.h.in
        ${gen_files_path}/module_config.h)

    # Copy additional sources
    file(COPY
        ${this_file_directory}/module.cpp
        ${this_file_directory}/pycompat.hpp
        ${this_file_directory}/py_cv_converters.hpp
        ${this_file_directory}/utils.hpp
        DESTINATION ${gen_files_path})

    # Module target -----------------------------------------------------------
    
    add_library(${MODULE_NAME} SHARED
        ${gen_files_path}/module.cpp
        ${add_python_opencv_bindings_UNPARSED_ARGUMENTS})

    add_dependencies(${MODULE_NAME} generated_headers)

    set_property(TARGET ${MODULE_NAME}
        PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE ON)
    target_compile_features(${MODULE_NAME} PRIVATE cxx_long_long_type)

    # Note: the module does not provide a header, the interface is brought by
    # internal symbols read by python (namely the PyInit_<modulename> function).
    target_include_directories(${MODULE_NAME} PRIVATE
        ${gen_files_path}
        ${PYTHON_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(${MODULE_NAME}
        ${PYTHON_LIBRARIES}
        ${OpenCV_LIBS})

    if(WIN32 OR CYGWIN)
        set(module_extension ".pyd")
    else()
        set(module_extension ".so")
    endif()
    set_target_properties(${MODULE_NAME}  PROPERTIES
        PREFIX ""
        EXTENSION ${module_extension})

endfunction()
