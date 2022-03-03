# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

set(TEST_RESOURCES_DIR ${CMAKE_SOURCE_DIR}/test/resources)
file(MAKE_DIRECTORY ${TEST_RESOURCES_DIR})
add_definitions (-DTEST_RESOURCE_DIR="${TEST_RESOURCES_DIR}")
set(TEST_TARGET_NAME "${CMAKE_PROJECT_NAME}-tests")

include(../common/cmake/find_catch.cmake)

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

ExternalProject_Add(ssd_mobile
        URL https://github.com/ARM-software/ML-zoo/raw/master/models/object_detection/ssd_mobilenet_v1/tflite_uint8/ssd_mobilenet_v1.tflite
        DOWNLOAD_NO_EXTRACT 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/ssd_mobilenet_v1.tflite ${CMAKE_CURRENT_SOURCE_DIR}/test/resources
        INSTALL_COMMAND ""
        )

ExternalProject_Add(yolo_v3
        URL https://github.com/ARM-software/ML-zoo/raw/master/models/object_detection/yolo_v3_tiny/tflite_fp32/yolo_v3_tiny_darknet_fp32.tflite
        DOWNLOAD_NO_EXTRACT 1
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ${CMAKE_COMMAND} -E copy <DOWNLOAD_DIR>/yolo_v3_tiny_darknet_fp32.tflite ${CMAKE_CURRENT_SOURCE_DIR}/test/resources
        INSTALL_COMMAND ""
        )

add_executable("${TEST_TARGET_NAME}" ${SOURCES} ${TEST_SOURCES} ${CVUTILS_SOURCES} ${UTILS_SOURCES})

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
    ${OPENCV_INCLUDE_DIR} ${DEPENDENCIES_DIR} ${TEST_RESOURCES_DIR} ${COMMON_INCLUDE_DIR})

target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${ARMNN_LIBS} ${OPENCV_LIBS} ${FFMPEG_LIBS})
if( USE_ARMNN_DELEGATE )
    set(CMAKE_CXX_FLAGS " -ldl -lrt -Wl,--copy-dt-needed-entries")
    target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${TfLite_LIB})
    target_link_libraries("${TEST_TARGET_NAME}" PUBLIC tflite_headers)
    target_include_directories("${TEST_TARGET_NAME}" PUBLIC ${Flatbuffers_INCLUDE_DIR})
    target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${Flatbuffers_LIB})
endif()