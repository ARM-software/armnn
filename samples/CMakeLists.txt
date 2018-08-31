if(BUILD_SAMPLE_APP)
    add_executable(SimpleSample SimpleSample.cpp)
    target_link_libraries(SimpleSample armnn pthread)
endif()
