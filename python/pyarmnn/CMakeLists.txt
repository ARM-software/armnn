#
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
#
set(SETUP_PY_IN         "${CMAKE_CURRENT_SOURCE_DIR}/setup.py")
set(SETUP_PY            "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
set(SWIG_GENERATE_IN    "${CMAKE_CURRENT_SOURCE_DIR}/swig_generate.py")
set(SWIG_GENERATE       "${CMAKE_CURRENT_BINARY_DIR}/swig_generate.py")
set(OUT_WRAP            "${CMAKE_CURRENT_BINARY_DIR}/pyarmnn.wrap.timestamp")

configure_file(${SETUP_PY_IN} ${SETUP_PY} COPYONLY)
configure_file(${SWIG_GENERATE_IN} ${SWIG_GENERATE} COPYONLY)

# local env variables passed down to the python scripts
# scripts can thus be used standalone
set(ARMNN_ENV ARMNN_INCLUDE="${PROJECT_SOURCE_DIR}/include:${PROJECT_SOURCE_DIR}/profiling/common/include"
              ARMNN_LIB=${PROJECT_BINARY_DIR}
              SWIG_EXECUTABLE=${SWIG_EXECUTABLE})

# common step - generates swig wrappers and builds the lib
add_custom_command(OUTPUT ${OUT_WRAP}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/README.md ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE ${CMAKE_CURRENT_BINARY_DIR}
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_BINARY_DIR}/src
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test ${CMAKE_CURRENT_BINARY_DIR}/test
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Clearing Python build ..."
                   COMMAND ${Python3_EXECUTABLE} ${SETUP_PY} --quiet clean --all
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Generating SWIG wrappers ..."
                   COMMAND ${CMAKE_COMMAND} -E env ${ARMNN_ENV} ${Python3_EXECUTABLE} ${SWIG_GENERATE}
                   COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Building Python extensions ..."
                   COMMAND ${CMAKE_COMMAND} -E env ${ARMNN_ENV} ${Python3_EXECUTABLE} ${SETUP_PY} --quiet build_ext --inplace
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/pyarmnn/_generated ${CMAKE_CURRENT_BINARY_DIR}/src/pyarmnn/_generated
                   COMMAND ${CMAKE_COMMAND} -E touch ${OUT_WRAP}
                   DEPENDS armnn)

# source package
if(BUILD_PYTHON_SRC)
    set(OUT_SRC "${CMAKE_CURRENT_BINARY_DIR}/pyarmnn.src.timestamp")
    add_custom_command(OUTPUT  ${OUT_SRC}
                       COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Building Python source package ..."
                       COMMAND ${CMAKE_COMMAND} -E env ${ARMNN_ENV} ${Python3_EXECUTABLE} ${SETUP_PY} --quiet sdist
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUT_SRC}
                       DEPENDS ${OUT_WRAP})
endif()
# wheel package
if(BUILD_PYTHON_WHL)
    set(OUT_WHL "${CMAKE_CURRENT_BINARY_DIR}/pyarmnn.whl.timestamp")
    add_custom_command(OUTPUT  ${OUT_WHL}
                       COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --green "Building Python binary package ..."
                       COMMAND ${CMAKE_COMMAND} -E env ${ARMNN_ENV} ${Python3_EXECUTABLE} ${SETUP_PY} --quiet bdist_wheel
                       COMMAND ${CMAKE_COMMAND} -E touch ${OUT_WHL}
                       DEPENDS ${OUT_WRAP})
endif()
add_custom_target(pyarmnn ALL DEPENDS ${OUT_WRAP} ${OUT_SRC} ${OUT_WHL})
