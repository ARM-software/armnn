# UnitTests
include(CheckIncludeFiles)

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
target_include_directories(inferenceTest PRIVATE ../src/backends)
target_include_directories(inferenceTest PRIVATE ../third-party/stb)

if (BUILD_TF_LITE_PARSER)
    macro(TfLiteParserTest testName sources)
        add_executable_ex(${testName} ${sources})
        target_include_directories(${testName} PRIVATE ../src/armnnUtils)
        target_include_directories(${testName} PRIVATE ../src/backends)

        target_link_libraries(${testName} inferenceTest)
        target_link_libraries(${testName} armnnTfLiteParser)
        target_link_libraries(${testName} armnn)
        target_link_libraries(${testName} ${CMAKE_THREAD_LIBS_INIT})
        addDllCopyCommands(${testName})
    endmacro()

    set(TfLiteBenchmark-Armnn_sources
        TfLiteBenchmark-Armnn/TfLiteBenchmark-Armnn.cpp)
    TfLiteParserTest(TfLiteBenchmark-Armnn "${TfLiteBenchmark-Armnn_sources}")

    set(TfLiteMobilenetQuantized-Armnn_sources
        TfLiteMobilenetQuantized-Armnn/TfLiteMobilenetQuantized-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteMobilenetQuantized-Armnn "${TfLiteMobilenetQuantized-Armnn_sources}")

    set(TfLiteMobileNetSsd-Armnn_sources
        TfLiteMobileNetSsd-Armnn/TfLiteMobileNetSsd-Armnn.cpp
        MobileNetSsdDatabase.hpp
        MobileNetSsdInferenceTest.hpp
        ObjectDetectionCommon.hpp)
    TfLiteParserTest(TfLiteMobileNetSsd-Armnn "${TfLiteMobileNetSsd-Armnn_sources}")

    set(TfLiteMobilenetV2Quantized-Armnn_sources
        TfLiteMobilenetV2Quantized-Armnn/TfLiteMobilenetV2Quantized-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteMobilenetV2Quantized-Armnn "${TfLiteMobilenetV2Quantized-Armnn_sources}")

    set(TfLiteVGG16Quantized-Armnn_sources
        TfLiteVGG16Quantized-Armnn/TfLiteVGG16Quantized-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteVGG16Quantized-Armnn "${TfLiteVGG16Quantized-Armnn_sources}")

    set(TfLiteMobileNetQuantizedSoftmax-Armnn_sources
        TfLiteMobileNetQuantizedSoftmax-Armnn/TfLiteMobileNetQuantizedSoftmax-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteMobileNetQuantizedSoftmax-Armnn "${TfLiteMobileNetQuantizedSoftmax-Armnn_sources}")

    set(TfLiteInceptionV3Quantized-Armnn_sources
        TfLiteInceptionV3Quantized-Armnn/TfLiteInceptionV3Quantized-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteInceptionV3Quantized-Armnn "${TfLiteInceptionV3Quantized-Armnn_sources}")

    set(TfLiteInceptionV4Quantized-Armnn_sources
        TfLiteInceptionV4Quantized-Armnn/TfLiteInceptionV4Quantized-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteInceptionV4Quantized-Armnn "${TfLiteInceptionV4Quantized-Armnn_sources}")

    set(TfLiteResNetV2-Armnn_sources
        TfLiteResNetV2-Armnn/TfLiteResNetV2-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteResNetV2-Armnn "${TfLiteResNetV2-Armnn_sources}")

    set(TfLiteResNetV2-50-Quantized-Armnn_sources
        TfLiteResNetV2-50-Quantized-Armnn/TfLiteResNetV2-50-Quantized-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteResNetV2-50-Quantized-Armnn "${TfLiteResNetV2-50-Quantized-Armnn_sources}")

    set(TfLiteMnasNet-Armnn_sources
        TfLiteMnasNet-Armnn/TfLiteMnasNet-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteMnasNet-Armnn "${TfLiteMnasNet-Armnn_sources}")


    set(TfLiteYoloV3Big-Armnn_sources
        TfLiteYoloV3Big-Armnn/NMS.cpp
        TfLiteYoloV3Big-Armnn/NMS.hpp
        TfLiteYoloV3Big-Armnn/TfLiteYoloV3Big-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfLiteParserTest(TfLiteYoloV3Big-Armnn "${TfLiteYoloV3Big-Armnn_sources}")



endif()

if (BUILD_ONNX_PARSER)
    macro(OnnxParserTest testName sources)
        add_executable_ex(${testName} ${sources})
        target_include_directories(${testName} PRIVATE ../src/armnnUtils)
        target_include_directories(${testName} PRIVATE ../src/backends)

        target_link_libraries(${testName} inferenceTest)
        target_link_libraries(${testName} armnnOnnxParser)
        target_link_libraries(${testName} armnn)
        target_link_libraries(${testName} ${CMAKE_THREAD_LIBS_INIT})
        addDllCopyCommands(${testName})
    endmacro()

    set(OnnxMnist-Armnn_sources
        OnnxMnist-Armnn/OnnxMnist-Armnn.cpp
        MnistDatabase.hpp
        MnistDatabase.cpp)
    OnnxParserTest(OnnxMnist-Armnn "${OnnxMnist-Armnn_sources}")

    set(OnnxMobileNet-Armnn_sources
        OnnxMobileNet-Armnn/OnnxMobileNet-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    OnnxParserTest(OnnxMobileNet-Armnn "${OnnxMobileNet-Armnn_sources}")
endif()

if (BUILD_ARMNN_SERIALIZER
        OR BUILD_TF_PARSER
        OR BUILD_TF_LITE_PARSER
        OR BUILD_ONNX_PARSER
        OR BUILD_ARMNN_TFLITE_DELEGATE)
    set(ExecuteNetwork_sources
        ExecuteNetwork/ExecuteNetwork.cpp
        ExecuteNetwork/ExecuteNetworkProgramOptions.cpp
        ExecuteNetwork/ExecuteNetworkProgramOptions.hpp
        ExecuteNetwork/ExecuteNetworkParams.cpp
        ExecuteNetwork/ExecuteNetworkParams.hpp
        NetworkExecutionUtils/NetworkExecutionUtils.cpp
        NetworkExecutionUtils/NetworkExecutionUtils.hpp)

    add_executable_ex(ExecuteNetwork ${ExecuteNetwork_sources})
    target_include_directories(ExecuteNetwork PRIVATE ../src/armnn)
    target_include_directories(ExecuteNetwork PRIVATE ../src/armnnUtils)
    target_include_directories(ExecuteNetwork PRIVATE ../src/backends)
    target_include_directories(ExecuteNetwork PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (BUILD_ARMNN_SERIALIZER)
        target_link_libraries(ExecuteNetwork armnnSerializer)
    endif()

    if (BUILD_TF_LITE_PARSER)
        target_link_libraries(ExecuteNetwork armnnTfLiteParser)
    endif()
    if (BUILD_ONNX_PARSER)
        target_link_libraries(ExecuteNetwork armnnOnnxParser)
    endif()
    if (BUILD_ARMNN_TFLITE_DELEGATE)
        target_link_libraries(ExecuteNetwork ArmnnDelegate::ArmnnDelegate)
    endif()
    target_link_libraries(ExecuteNetwork armnn)

   target_link_libraries(ExecuteNetwork ${CMAKE_THREAD_LIBS_INIT})
   addDllCopyCommands(ExecuteNetwork)
endif()

if(BUILD_ACCURACY_TOOL)
    macro(AccuracyTool executorName)
        target_link_libraries(${executorName} ${CMAKE_THREAD_LIBS_INIT})
        if (BUILD_ARMNN_SERIALIZER)
            target_link_libraries(${executorName} armnnSerializer)
        endif()
        if (BUILD_TF_LITE_PARSER)
            target_link_libraries(${executorName} armnnTfLiteParser)
        endif()
        if (BUILD_ONNX_PARSER)
            target_link_libraries(${executorName} armnnOnnxParser)
        endif()
        addDllCopyCommands(${executorName})
    endmacro()

    set(ModelAccuracyTool-Armnn_sources
            ModelAccuracyTool-Armnn/ModelAccuracyTool-Armnn.cpp)

    add_executable_ex(ModelAccuracyTool ${ModelAccuracyTool-Armnn_sources})
    target_include_directories(ModelAccuracyTool PRIVATE ../src/armnn)
    target_include_directories(ModelAccuracyTool PRIVATE ../src/armnnUtils)
    target_include_directories(ModelAccuracyTool PRIVATE ../src/backends)
    target_link_libraries(ModelAccuracyTool inferenceTest)
    target_link_libraries(ModelAccuracyTool armnn)
    target_link_libraries(ModelAccuracyTool armnnSerializer)
    AccuracyTool(ModelAccuracyTool)
endif()

if(BUILD_ACCURACY_TOOL)
    macro(ImageTensorExecutor executorName)
        target_link_libraries(${executorName} ${CMAKE_THREAD_LIBS_INIT})
        addDllCopyCommands(${executorName})
    endmacro()

    set(ImageTensorGenerator_sources
            InferenceTestImage.hpp
            InferenceTestImage.cpp
            ImageTensorGenerator/ImageTensorGenerator.cpp)

    add_executable_ex(ImageTensorGenerator ${ImageTensorGenerator_sources})
    target_include_directories(ImageTensorGenerator PRIVATE ../src/armnn)
    target_include_directories(ImageTensorGenerator PRIVATE ../src/armnnUtils)

    target_link_libraries(ImageTensorGenerator armnn)
    ImageTensorExecutor(ImageTensorGenerator)

    set(ImageCSVFileGenerator_sources
            ImageCSVFileGenerator/ImageCSVFileGenerator.cpp)

    add_executable_ex(ImageCSVFileGenerator ${ImageCSVFileGenerator_sources})
    target_include_directories(ImageCSVFileGenerator PRIVATE ../src/armnnUtils)
    ImageTensorExecutor(ImageCSVFileGenerator)
endif()

if(BUILD_MEMORY_STRATEGY_BENCHMARK)
    add_subdirectory(MemoryStrategyBenchmark)
endif()
