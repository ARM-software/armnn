#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(BUILD_TF_LITE_PARSER)
    set(armnn_tf_lite_parser_sources)
    list(APPEND armnn_tf_lite_parser_sources
        ../../include/armnnTfLiteParser/ITfLiteParser.hpp
        ../../include/armnnTfLiteParser/Version.hpp
        TfLiteParser.hpp
        TfLiteParser.cpp
        )

    add_library_ex(armnnTfLiteParser SHARED ${armnn_tf_lite_parser_sources})

    # NOTE: even though the tensorflow sources contain a ./tensorflow/lite/schema/schema_generated.h
    #       file we cannot use this directly because we need to take packaging for linux distros into
    #       account. On Ubuntu 20.04 the available package is flatbuffers 1.11 and on 21.10 it is 1.12
    #       despite the minor versioning they are not backward compatible. The current tensorflow lite
    #       source (v2.3-v2.5) is generated from 1.12... so we need to generate from the
    #       ./tensorflow/lite/schema/schema.fbs in the tensorflow lite source using the flatc matching
    #       the target platform but use the ./tensorflow/lite/version.h to determine which version of
    #       tensorflow lite the header was generated from.
    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")
    set_target_properties(armnnTfLiteParser PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_include_directories(armnnTfLiteParser PRIVATE ../armnn)
    target_include_directories(armnnTfLiteParser PRIVATE ../armnnUtils)
    target_include_directories(armnnTfLiteParser SYSTEM PRIVATE "${TF_LITE_SCHEMA_INCLUDE_PATH}")


    # using the armnn/delegate/cmake/Modules/FindTfLite.cmake to find the TfLite sources
    # so that we can use the tensorflow/lite/version.h to determine which version of
    # tensorflow lite we are compiling against
    find_package(TfLite REQUIRED MODULE)

    # Various tflite header files are not warning clean
    # We can't change compilation flags on header files directly, so we need to add them to an interface library first
    add_library(tflite_version_headers INTERFACE)
    target_include_directories(tflite_version_headers INTERFACE $<BUILD_INTERFACE:${TfLite_INCLUDE_DIR}>
                                                                $<INSTALL_INTERFACE:include/tflite_version_headers>)

    target_compile_options(tflite_version_headers INTERFACE -Wno-conversion
                                                            -Wno-sign-conversion
                                                            -Wno-unused-parameter
                                                            -Wno-unused-function)

    # If user has explicitly specified flatbuffers lib then use that,
    # otherwise search for it based on FLATBUFFERS_BUILD_DIR
    if (FLATBUFFERS_LIBRARY)
      target_link_libraries(armnnTfLiteParser
                            armnn
                            tflite_version_headers
                            ${FLATBUFFERS_LIBRARY})
    else()
        # Use PATH_SUFFIXES to help find separate libs for debug/release on Windows builds
        find_library(FLATBUFFERS_LIBRARY_DEBUG NAMES flatbuffers
                     HINTS ${FLATBUFFERS_BUILD_DIR}
                     PATH_SUFFIXES "Debug")
        find_library(FLATBUFFERS_LIBRARY_RELEASE NAMES flatbuffers
                     HINTS ${FLATBUFFERS_BUILD_DIR}
                     PATH_SUFFIXES "Release")
                   target_link_libraries(armnnTfLiteParser
                                         armnn
                                         tflite_version_headers
                                         debug ${FLATBUFFERS_LIBRARY_DEBUG}
                                         optimized ${FLATBUFFERS_LIBRARY_RELEASE})
    endif()

    set_target_properties(armnnTfLiteParser PROPERTIES VERSION ${TFLITE_PARSER_LIB_VERSION} SOVERSION ${TFLITE_PARSER_LIB_SOVERSION} )

    install(TARGETS armnnTfLiteParser
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
