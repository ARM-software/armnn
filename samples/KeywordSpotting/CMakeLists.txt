# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
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

project (keyword-spotting-example)

set(CMAKE_C_FLAGS_DEBUG         "-DDEBUG -O0 -g -fPIC -pthread")
set(CMAKE_C_FLAGS_RELEASE       "-DNDEBUG -O3 -fPIC -pthread")

set(CMAKE_CXX_FLAGS_DEBUG       "-DDEBUG -O0 -g -fPIC -pthread")
set(CMAKE_CXX_FLAGS_RELEASE     "-DNDEBUG -O3 -fPIC -pthread")

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

include(../common/cmake/find_armnn.cmake)

include_directories(include)
include_directories(../common/include/ArmnnUtils)
include_directories(../common/include/Utils)
include_directories(../common/include/Audio)

file(GLOB SOURCES "src/*.cpp")
file(GLOB COMMON_UTILS_SOURCES "../common/src/Utils/*.cpp")
file(GLOB COMMON_AUDIO_SOURCES "../common/src/Audio/*.cpp")
list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/Main.cpp)
file(GLOB TEST_SOURCES "test/*.cpp")
file(GLOB APP_MAIN "src/Main.cpp")

if(BUILD_UNIT_TESTS)
    include(cmake/unit_tests.cmake)
endif()

set(APP_TARGET_NAME "${CMAKE_PROJECT_NAME}")

add_executable("${APP_TARGET_NAME}"  ${COMMON_UTILS_SOURCES} ${COMMON_AUDIO_SOURCES} ${SOURCES} ${APP_MAIN})

target_link_libraries("${APP_TARGET_NAME}" PUBLIC ${ARMNN_LIBS} -lsndfile -lsamplerate)
target_include_directories("${APP_TARGET_NAME}" PUBLIC ${ARMNN_INCLUDE_DIR} )
