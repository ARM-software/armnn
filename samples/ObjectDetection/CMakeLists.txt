# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.0.2)

set(CMAKE_C_STANDARD                99)
set(CMAKE_CXX_STANDARD              14)

# Make the standard a requirement => prevent fallback to previous
# supported standard
set(CMAKE_C_STANDARD_REQUIRED       ON)
set(CMAKE_CXX_STANDARD_REQUIRED     ON)

# We want to pass standard C/C++ flags, without gnu extensions
set(CMAKE_C_EXTENSIONS              OFF)
set(CMAKE_CXX_EXTENSIONS            OFF)

project (object_detection_example)

set(CMAKE_C_FLAGS_DEBUG         "-DDEBUG -O0 -g -fPIC")
set(CMAKE_C_FLAGS_RELEASE       "-DNDEBUG -O3 -fPIC")

set(CMAKE_CXX_FLAGS_DEBUG       "-DDEBUG -O0 -g -fPIC")
set(CMAKE_CXX_FLAGS_RELEASE     "-DNDEBUG -O3 -fPIC")

include(ExternalProject)

# Build in release mode by default
if (NOT CMAKE_BUILD_TYPE STREQUAL Debug)
    set(CMAKE_BUILD_TYPE Release CACHE INTERNAL "")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (NOT DEFINED DEPENDENCIES_DIR)
    set(DEPENDENCIES_DIR ${CMAKE_BINARY_DIR}/dependencies)
endif()

include(../common/cmake/find_opencv.cmake)
include(../common/cmake/find_armnn.cmake)

include_directories(include)
include_directories(../common/include/ArmnnUtils)
include_directories(../common/include/Utils)
include_directories(../common/include/CVUtils)

file(GLOB SOURCES "src/*.cpp")
file(GLOB CVUTILS_SOURCES "../common/src/CVUtils**/*.cpp")
file(GLOB UTILS_SOURCES "../common/src/Utils**/*.cpp")
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)
file(GLOB TEST_SOURCES "test/*.cpp")
file(GLOB APP_MAIN "src/Main.cpp")

if(BUILD_UNIT_TESTS)
    include(cmake/unit_tests.cmake)
endif()


set(APP_TARGET_NAME "${CMAKE_PROJECT_NAME}")

add_executable("${APP_TARGET_NAME}" ${SOURCES} ${CVUTILS_SOURCES} ${UTILS_SOURCES} ${APP_MAIN})

if (NOT OPENCV_LIBS_FOUND)
    message("Building OpenCV libs")
    add_dependencies("${APP_TARGET_NAME}" "${OPENCV_LIB}")
endif()

target_link_libraries("${APP_TARGET_NAME}" PUBLIC ${ARMNN_LIBS} ${OPENCV_LIBS})
target_include_directories("${APP_TARGET_NAME}" PUBLIC ${ARMNN_INCLUDE_DIR} ${OPENCV_INCLUDE_DIR})
