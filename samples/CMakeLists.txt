if(BUILD_SAMPLE_APP AND ARMNNREF)
    add_executable(SimpleSample SimpleSample.cpp)
    target_link_libraries(SimpleSample armnn ${CMAKE_THREAD_LIBS_INIT})

    add_executable(AsyncExecutionSample AsyncExecutionSample.cpp)
    target_link_libraries(AsyncExecutionSample armnn ${CMAKE_THREAD_LIBS_INIT})
endif()

if(BUILD_SAMPLE_APP AND SAMPLE_DYNAMIC_BACKEND)
    add_executable(DynamicSample DynamicSample.cpp)
    target_link_libraries(DynamicSample armnn ${CMAKE_THREAD_LIBS_INIT})
endif()

if(BUILD_SAMPLE_APP AND ARMCOMPUTECL)
    add_executable(CustomMemoryAllocatorSample CustomMemoryAllocatorSample.cpp)
    target_link_libraries(CustomMemoryAllocatorSample armnn ${CMAKE_THREAD_LIBS_INIT})
endif()

if(BUILD_SAMPLE_APP AND ARMNNREF)
    add_executable(PreImportMemorySample PreImportMemorySample.cpp)
    target_link_libraries(PreImportMemorySample armnn ${CMAKE_THREAD_LIBS_INIT})
endif()

