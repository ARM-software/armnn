#
# Copyright © 2017, 2019-2020, 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(BUILD_ARMNN_SERIALIZER)
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
        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../../generated/ArmnnSchema_generated.h DEPENDS ArmnnSchema.fbs
        COMMAND ${FLATC} -o ${CMAKE_CURRENT_SOURCE_DIR}/../../generated --cpp ${CMAKE_CURRENT_SOURCE_DIR}/ArmnnSchema.fbs
    )
    set_property(SOURCE Serializer.cpp APPEND PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../generated/ArmnnSchema_generated.h)
    set_directory_properties(PROPERTIES ADDITIONAL_CLEAN_FILES ${CMAKE_CURRENT_SOURCE_DIR}/../../generated/)

    set(armnn_serializer_sources)
    list(APPEND armnn_serializer_sources
        ../../include/armnnSerializer/ISerializer.hpp
        ../../include/armnnDeserializer/IDeserializer.hpp
        Serializer.hpp
        Serializer.cpp
        SerializerUtils.hpp
        SerializerUtils.cpp
        ../armnnDeserializer/Deserializer.hpp
        ../armnnDeserializer/Deserializer.cpp
        )

    if(BUILD_BARE_METAL OR EXECUTE_NETWORK_STATIC)
        add_library_ex(armnnSerializer STATIC ${armnn_serializer_sources})
    else()
        # We're going to export both a STATIC library and a SHARED library here.
        # In some instances it's easier to include the serializer directly into
        # the target executable or library rather than have yet another .so.
        add_library(armnnSerializer-static STATIC ${armnn_serializer_sources})
        add_library_ex(armnnSerializer SHARED ${armnn_serializer_sources})
    endif()

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")

    set_target_properties(armnnSerializer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_include_directories(armnnSerializer PRIVATE ../armnn)
    target_include_directories(armnnSerializer PRIVATE ../armnnUtils)
    target_include_directories(armnnSerializer PRIVATE ../../generated)
    if (NOT BARE_METAL AND NOT EXECUTE_NETWORK_STATIC)
        target_include_directories(armnnSerializer-static PRIVATE ../armnn)
        target_include_directories(armnnSerializer-static PRIVATE ../armnnUtils)
        target_include_directories(armnnSerializer-static PRIVATE ../../generated)
    endif()

    list(APPEND armnn_serializer_sources
        ArmnnSchema_generated.h
        )

    # System include to suppress warnings for flatbuffers generated files
    target_include_directories(armnnSerializer SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_link_libraries(armnnSerializer armnn ${FLATBUFFERS_LIBRARY})
    if (NOT BARE_METAL AND NOT EXECUTE_NETWORK_STATIC)
        install(TARGETS armnnSerializer-static
                EXPORT  armnn-targets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
    endif()

    install(TARGETS armnnSerializer
            EXPORT  armnn-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    set_target_properties(armnnSerializer PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )
endif()
