#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(BUILD_ARMNN_SERIALIZER)
    find_program(FLATC flatc
                 HINTS ${FLATC_DIR}
                 DOC "Path to 'flatc', the flatbuffers compiler")
    if (NOT FLATC)
        message(SEND_ERROR "flatc not found. Specify the full path of the flatc executable with -DFLATC=<flatc path>")
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

    if(BUILD_BARE_METAL)
        add_library_ex(armnnSerializer STATIC ${armnn_serializer_sources})
    else()
        # We're going to export both an OBJECT library and a SHARED library here.
        # In some instances it's easier to include the serializer directly into
        # the target executable or library rather than have yet another .so.
        add_library(armnnSerializerObj OBJECT ${armnn_serializer_sources})
        add_library_ex(armnnSerializer SHARED ${armnn_serializer_sources})
    endif()

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")

    set_target_properties(armnnSerializer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_include_directories(armnnSerializerObj PRIVATE ../armnn)
    target_include_directories(armnnSerializerObj PRIVATE ../armnnUtils)
    target_include_directories(armnnSerializer PRIVATE ../armnn)
    target_include_directories(armnnSerializer PRIVATE ../armnnUtils)
    target_include_directories(armnnSerializer PRIVATE ../../generated)
    target_include_directories(armnnSerializerObj PRIVATE ../../generated)

    list(APPEND armnn_serializer_sources
        ArmnnSchema_generated.h
        )

    # System include to suppress warnings for flatbuffers generated files
    target_include_directories(armnnSerializerObj SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_include_directories(armnnSerializer SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_link_libraries(armnnSerializer armnn ${FLATBUFFERS_LIBRARY})

    install(TARGETS armnnSerializerObj
            EXPORT  armnn-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )

    install(TARGETS armnnSerializer
            EXPORT  armnn-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    set_target_properties(armnnSerializer PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )
endif()
