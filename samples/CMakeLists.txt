if(BUILD_SAMPLE_APP AND ARMCOMPUTEREF)
    add_executable(SimpleSample SimpleSample.cpp)
    target_link_libraries(SimpleSample armnn ${CMAKE_THREAD_LIBS_INIT})
endif()
