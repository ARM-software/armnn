if(BUILD_SAMPLE_APP AND ARMNNREF)
    add_executable(SimpleSample SimpleSample.cpp)
    target_link_libraries(SimpleSample armnn ${CMAKE_THREAD_LIBS_INIT})
endif()
