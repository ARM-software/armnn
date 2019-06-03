#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
if(BUILD_ARMNN_SERIALIZER)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/ArmnnSchema_generated.h)
    add_custom_command (
        OUTPUT  ArmnnSchema_generated.h
        COMMAND ${FLATC_DIR}/flatc -o ${CMAKE_CURRENT_BINARY_DIR} --cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/ArmnnSchema.fbs
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
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
