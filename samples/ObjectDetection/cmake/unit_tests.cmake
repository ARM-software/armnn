# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

set(TEST_RESOURCES_DIR ${CMAKE_SOURCE_DIR}/test/resources)
add_definitions (-DTEST_RESOURCE_DIR="${TEST_RESOURCES_DIR}")
set(TEST_TARGET_NAME "${CMAKE_PROJECT_NAME}-tests")

file(GLOB TEST_SOURCES "test/*")

include(cmake/find_catch.cmake)

file(DOWNLOAD "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
        ${CMAKE_CURRENT_SOURCE_DIR}/test/resources/models.zip SHOW_PROGRESS)

# Extract
execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xf models.zip
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test/resources/
        RESULT_VARIABLE return_code
)

ExternalProject_Add(basketball-image
        URL https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/basketball1.png
        DOWNLOAD_NO_EXTRACT 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/basketball1.png ${CMAKE_CURRENT_SOURCE_DIR}/test/resources
        INSTALL_COMMAND ""
)

ExternalProject_Add(messi
        URL https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/messi5.jpg
        DOWNLOAD_NO_EXTRACT 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/messi5.jpg ${CMAKE_CURRENT_SOURCE_DIR}/test/resources
        INSTALL_COMMAND ""
        )

ExternalProject_Add(vtest
        URL https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/Megamind.avi
        DOWNLOAD_NO_EXTRACT 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/Megamind.avi ${CMAKE_CURRENT_SOURCE_DIR}/test/resources
        INSTALL_COMMAND ""
        )

add_executable("${TEST_TARGET_NAME}" ${SOURCES} ${TEST_SOURCES})

add_dependencies(
    "${TEST_TARGET_NAME}"
    "catch2-headers"
    "vtest"
    "messi"
    "basketball-image"
)

if (NOT OPENCV_LIBS_FOUND)
    message("Building OpenCV libs")
    add_dependencies("${TEST_TARGET_NAME}" "${OPENCV_LIB}")
endif()

target_include_directories("${TEST_TARGET_NAME}" PUBLIC ${TEST_TPIP_INCLUDE}
    ${ARMNN_INCLUDE_DIR}
    ${OPENCV_INCLUDE_DIR} ${DEPENDENCIES_DIR} ${TEST_RESOURCES_DIR})

target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${ARMNN_LIBS} ${OPENCV_LIBS} ${FFMPEG_LIBS})