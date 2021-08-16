# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

set(TEST_RESOURCES_DIR ${CMAKE_SOURCE_DIR}/test/resources)
file(MAKE_DIRECTORY ${TEST_RESOURCES_DIR})
add_definitions (-DTEST_RESOURCE_DIR="${TEST_RESOURCES_DIR}")
set(TEST_TARGET_NAME "${CMAKE_PROJECT_NAME}-tests")

file(GLOB TEST_SOURCES "test/*")
file(GLOB TESTS_AUDIO_COMMON "../common/test/Audio/*")

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/resources)
include(../common/cmake/find_catch.cmake)

add_executable("${TEST_TARGET_NAME}" ${COMMON_UTILS_SOURCES} ${COMMON_AUDIO_SOURCES} ${SOURCES} ${TEST_SOURCES} ${TESTS_AUDIO_COMMON})

ExternalProject_Add(passport
        URL https://raw.githubusercontent.com/Azure-Samples/cognitive-services-speech-sdk/master/sampledata/audiofiles/myVoiceIsMyPassportVerifyMe04.wav
        DOWNLOAD_NO_EXTRACT 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/myVoiceIsMyPassportVerifyMe04.wav ${CMAKE_CURRENT_SOURCE_DIR}/test/resources
        INSTALL_COMMAND ""
        )

add_dependencies(
        "${TEST_TARGET_NAME}"
        "passport"
        "catch2-headers"
)

target_include_directories("${TEST_TARGET_NAME}" PUBLIC ${TEST_TPIP_INCLUDE}
    ${ARMNN_INCLUDE_DIR}
     ${DEPENDENCIES_DIR} ${TEST_RESOURCES_DIR} ${COMMON_INCLUDE_DIR})

target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${ARMNN_LIBS} -lsndfile -lsamplerate)