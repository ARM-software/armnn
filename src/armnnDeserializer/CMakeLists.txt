#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(BUILD_ARMNN_DESERIALIZER)
    # We're using NO_SYSTEM_ENVIRONMENT_PATH to prevent a system installed
    # flatc being found and used.
    find_program(FLATC flatc
                 HINTS ${FLATC_DIR}
                 NO_SYSTEM_ENVIRONMENT_PATH
                 DOC "Path to 'flatc', the flatbuffers compiler")

    if (NOT FLATC)
        message(SEND_ERROR "flatc not found. Specify the full path of the flatc executable with -DFLATC=<flatc path>")
    else()
        message("Using flatc was from: ${FLATC}")
    endif()

    add_custom_command(
        # Generate an ArmnnSchema_generated.h file if it doesn't exist, or update it when necessary otherwise
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../generated/ArmnnSchema_generated.h DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../armnnSerializer/ArmnnSchema.fbs
        COMMAND ${FLATC} -o ${CMAKE_CURRENT_SOURCE_DIR}/../../generated --cpp ${CMAKE_CURRENT_SOURCE_DIR}/../armnnSerializer/ArmnnSchema.fbs
    )
    set_property(SOURCE Deserializer.cpp APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../generated/ArmnnSchema_generated.h)
    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../generated/)

    set(armnn_deserializer_sources)
    list(APPEND armnn_deserializer_sources
            Deserializer.hpp
            Deserializer.cpp
        )

    add_library_ex(armnnDeserializer SHARED ${armnn_deserializer_sources})

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")

    set_target_properties(armnnDeserializer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_include_directories(armnnDeserializer PRIVATE ../armnn)
    target_include_directories(armnnDeserializer PRIVATE ../armnnUtils)
    target_include_directories(armnnDeserializer PRIVATE ../../generated)

    list(APPEND armnn_deserializer_sources
            ArmnnSchema_generated.h
        )

    # System include to suppress warnings for flatbuffers generated files
    target_include_directories(armnnDeserializer SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_link_libraries(armnnDeserializer armnn ${FLATBUFFERS_LIBRARY})

    install(TARGETS armnnDeserializer
            EXPORT armnn-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    set_target_properties(armnnDeserializer PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )
endif()
