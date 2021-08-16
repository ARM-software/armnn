# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

# Function to download a file from the Arm Model Zoo
function(download_file_from_modelzoo model_zoo_version file_sub_path download_path)
    set(MODEL_ZOO_REPO "https://github.com/ARM-software/ML-zoo/raw")
    string(JOIN "/" FILE_URL
        ${MODEL_ZOO_REPO} ${model_zoo_version} ${file_sub_path})
    message(STATUS "Downloading ${FILE_URL} to ${download_path}...")
    file(DOWNLOAD ${FILE_URL} ${download_path}
        STATUS DOWNLOAD_STATE)
    list(GET DOWNLOAD_STATE 0 RET_VAL)
    if(${RET_VAL})
        list(GET DOWNLOAD_STATE 1 RET_MSG)
        message(FATAL_ERROR "Download failed with error code: ${RET_VAL}; "
                            "Error message: ${RET_MSG}")
    endif()
endfunction()

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


set(MODEL_FILENAME          ds_cnn_clustered_int8.tflite)
set(MODEL_RESOURCES_DIR     ${CMAKE_CURRENT_SOURCE_DIR}/test/resources)
file(MAKE_DIRECTORY         ${MODEL_RESOURCES_DIR})
set(DEFAULT_MODEL_PATH      ${CMAKE_CURRENT_SOURCE_DIR}/test/resources/${MODEL_FILENAME})

# Download the default model
set(ZOO_COMMON_SUBPATH      "models/keyword_spotting/ds_cnn_large/tflite_clustered_int8")
set(ZOO_MODEL_SUBPATH       "${ZOO_COMMON_SUBPATH}/${MODEL_FILENAME}")
set(ZOO_MODEL_VERSION       "68b5fbc77ed28e67b2efc915997ea4477c1d9d5b")

download_file_from_modelzoo(${ZOO_MODEL_VERSION} ${ZOO_MODEL_SUBPATH} ${DEFAULT_MODEL_PATH})


target_include_directories("${TEST_TARGET_NAME}" PUBLIC ${TEST_TPIP_INCLUDE}
     ${ARMNN_INCLUDE_DIR}
      ${DEPENDENCIES_DIR} ${TEST_RESOURCES_DIR} ${COMMON_INCLUDE_DIR})

target_link_libraries("${TEST_TARGET_NAME}" PUBLIC ${ARMNN_LIBS} -lsndfile -lsamplerate)