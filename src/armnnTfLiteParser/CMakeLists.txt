#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(BUILD_TF_LITE_PARSER)
    set(armnn_tf_lite_parser_sources)
    list(APPEND armnn_tf_lite_parser_sources
        ../../include/armnnTfLiteParser/ITfLiteParser.hpp
        TfLiteParser.hpp
        TfLiteParser.cpp
        )

    add_library_ex(armnnTfLiteParser SHARED ${armnn_tf_lite_parser_sources})

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")
    set_target_properties(armnnTfLiteParser PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_include_directories(armnnTfLiteParser PRIVATE ../armnn)
    target_include_directories(armnnTfLiteParser PRIVATE ../armnnUtils)
    target_include_directories(armnnTfLiteParser SYSTEM PRIVATE "${TF_LITE_SCHEMA_INCLUDE_PATH}")

    target_link_libraries(armnnTfLiteParser ${Boost_FILESYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
    target_link_libraries(armnnTfLiteParser armnn ${FLATBUFFERS_LIBRARY})

    install(TARGETS armnnTfLiteParser
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
