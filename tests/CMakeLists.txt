# UnitTests
include(CheckIncludeFiles)

include_directories(SYSTEM "${THIRD_PARTY_INCLUDE_DIRS}")

# Setup the inference test framework
set(inference_test_sources
    ClassifierTestCaseData.hpp
    InferenceModel.hpp
    InferenceTest.hpp
    InferenceTest.inl
    InferenceTest.cpp
    InferenceTestImage.hpp
    InferenceTestImage.cpp)
add_library_ex(inferenceTest STATIC ${inference_test_sources})
target_include_directories(inferenceTest PRIVATE ../src/armnnUtils)

if(BUILD_CAFFE_PARSER)
    macro(CaffeParserTest testName sources)
        add_executable_ex(${testName} ${sources})
        target_include_directories(${testName} PRIVATE ../src/armnnUtils)
        set_target_properties(${testName} PROPERTIES COMPILE_FLAGS "${CAFFE_PARSER_TEST_ADDITIONAL_COMPILE_FLAGS}")

        target_link_libraries(${testName} inferenceTest)
        target_link_libraries(${testName} armnnCaffeParser)
        target_link_libraries(${testName} armnn)
        target_link_libraries(${testName} ${CMAKE_THREAD_LIBS_INIT})
        if(OPENCL_LIBRARIES)
            target_link_libraries(${testName} ${OPENCL_LIBRARIES})
        endif()
        target_link_libraries(${testName}
            ${Boost_SYSTEM_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_PROGRAM_OPTIONS_LIBRARY})
        addDllCopyCommands(${testName})
    endmacro()

    set(CaffeCifar10AcrossChannels-Armnn_sources
        CaffeCifar10AcrossChannels-Armnn/CaffeCifar10AcrossChannels-Armnn.cpp
        Cifar10Database.hpp
        Cifar10Database.cpp)
    CaffeParserTest(CaffeCifar10AcrossChannels-Armnn "${CaffeCifar10AcrossChannels-Armnn_sources}")

    set(CaffeMnist-Armnn_sources
        CaffeMnist-Armnn/CaffeMnist-Armnn.cpp
        MnistDatabase.hpp
        MnistDatabase.cpp)
    CaffeParserTest(CaffeMnist-Armnn "${CaffeMnist-Armnn_sources}")

    set(CaffeAlexNet-Armnn_sources
        CaffeAlexNet-Armnn/CaffeAlexNet-Armnn.cpp
        ImageNetDatabase.hpp
        ImageNetDatabase.cpp)
    CaffeParserTest(CaffeAlexNet-Armnn "${CaffeAlexNet-Armnn_sources}")

    set(MultipleNetworksCifar10_SRC
        MultipleNetworksCifar10/MultipleNetworksCifar10.cpp
        Cifar10Database.hpp
        Cifar10Database.cpp)
    CaffeParserTest(MultipleNetworksCifar10 "${MultipleNetworksCifar10_SRC}")

    set(CaffeResNet-Armnn_sources
        CaffeResNet-Armnn/CaffeResNet-Armnn.cpp
        ImageNetDatabase.hpp
        ImageNetDatabase.cpp)
    CaffeParserTest(CaffeResNet-Armnn "${CaffeResNet-Armnn_sources}")

    set(CaffeVGG-Armnn_sources
        CaffeVGG-Armnn/CaffeVGG-Armnn.cpp
        ImageNetDatabase.hpp
        ImageNetDatabase.cpp)
    CaffeParserTest(CaffeVGG-Armnn "${CaffeVGG-Armnn_sources}")

    set(CaffeInception_BN-Armnn_sources
        CaffeInception_BN-Armnn/CaffeInception_BN-Armnn.cpp
        ImageNetDatabase.hpp
        ImageNetDatabase.cpp)
    CaffeParserTest(CaffeInception_BN-Armnn "${CaffeInception_BN-Armnn_sources}")

    set(CaffeYolo-Armnn_sources
        CaffeYolo-Armnn/CaffeYolo-Armnn.cpp
        YoloDatabase.hpp
        YoloDatabase.cpp
        YoloInferenceTest.hpp)
    CaffeParserTest(CaffeYolo-Armnn "${CaffeYolo-Armnn_sources}")
endif()

if (BUILD_CAFFE_PARSER)
    set(ExecuteNetwork_sources
        ExecuteNetwork/ExecuteNetwork.cpp)

    add_executable_ex(ExecuteNetwork ${ExecuteNetwork_sources})
    target_include_directories(ExecuteNetwork PRIVATE ../src/armnnUtils)

    if (BUILD_CAFFE_PARSER)
        target_link_libraries(ExecuteNetwork armnnCaffeParser)
    endif()
    target_link_libraries(ExecuteNetwork armnn)
    target_link_libraries(ExecuteNetwork ${CMAKE_THREAD_LIBS_INIT})
    if(OPENCL_LIBRARIES)
        target_link_libraries(ExecuteNetwork ${OPENCL_LIBRARIES})
    endif()
    target_link_libraries(ExecuteNetwork
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY})
    addDllCopyCommands(ExecuteNetwork)
endif()
