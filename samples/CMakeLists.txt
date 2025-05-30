#
# Copyright © 2018-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#
if (NOT BUILD_BARE_METAL)
   if(BUILD_SAMPLE_APP AND ARMNNREF)
        add_executable(SimpleSample SimpleSample.cpp)
        target_link_libraries(SimpleSample armnn ${CMAKE_THREAD_LIBS_INIT})
    endif()

    if(BUILD_SAMPLE_APP AND SAMPLE_DYNAMIC_BACKEND)
        add_executable(DynamicSample DynamicSample.cpp)
        target_link_libraries(DynamicSample armnn ${CMAKE_THREAD_LIBS_INIT})
    endif()

    if(BUILD_SAMPLE_APP AND ARMCOMPUTECL)
        add_executable(CustomMemoryAllocatorSample CustomMemoryAllocatorSample.cpp)
        target_link_libraries(CustomMemoryAllocatorSample armnn ${CMAKE_THREAD_LIBS_INIT})
    endif()
else()
    if (ARMNNREF)
        add_executable(BareMetalDeserializedGraph BareMetalDeserializedGraph.cpp)
        target_link_libraries(BareMetalDeserializedGraph fmt pipeCommon pipeClient armnn)
    endif()
endif()
