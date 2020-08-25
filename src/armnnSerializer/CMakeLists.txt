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
        OUTPUT ArmnnSchema_generated.h DEPENDS ArmnnSchema.fbs
        COMMAND ${FLATC} -o ${CMAKE_CURRENT_BINARY_DIR} --cpp ${CMAKE_CURRENT_SOURCE_DIR}/ArmnnSchema.fbs
    )

    set(armnn_serializer_sources)
    list(APPEND armnn_serializer_sources
        ../../include/armnnSerializer/ISerializer.hpp
        ../../include/armnnDeserializer/IDeserializer.hpp
        ArmnnSchema_generated.h
        Serializer.hpp
        Serializer.cpp
        SerializerUtils.hpp
        SerializerUtils.cpp
        ../armnnDeserializer/Deserializer.hpp
        ../armnnDeserializer/Deserializer.cpp
        )

    add_library_ex(armnnSerializer SHARED ${armnn_serializer_sources})

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")
    set_target_properties(armnnSerializer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
    target_include_directories(armnnSerializer PRIVATE ../armnn)
    target_include_directories(armnnSerializer PRIVATE ../armnnUtils)

    # System include to suppress warnings for flatbuffers generated files
    target_include_directories(armnnSerializer SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

    target_link_libraries(armnnSerializer armnn ${FLATBUFFERS_LIBRARY})

    install(TARGETS armnnSerializer
            EXPORT  armnn-targets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    set_target_properties(armnnSerializer PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION} )
endif()
