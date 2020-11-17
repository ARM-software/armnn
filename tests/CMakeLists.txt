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

if(BUILD_CAFFE_PARSER)
    macro(CaffeParserTest testName sources)
        add_executable_ex(${testName} ${sources})
        target_include_directories(${testName} PRIVATE ../src/armnnUtils)
        target_include_directories(${testName} PRIVATE ../src/backends)
        set_target_properties(${testName} PROPERTIES COMPILE_FLAGS "${CAFFE_PARSER_TEST_ADDITIONAL_COMPILE_FLAGS}")

        target_link_libraries(${testName} inferenceTest)
        target_link_libraries(${testName} armnnCaffeParser)
        target_link_libraries(${testName} armnn)
        target_link_libraries(${testName} ${CMAKE_THREAD_LIBS_INIT})
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
        CaffePreprocessor.hpp
        CaffePreprocessor.cpp)
    CaffeParserTest(CaffeAlexNet-Armnn "${CaffeAlexNet-Armnn_sources}")

    set(MultipleNetworksCifar10_SRC
        MultipleNetworksCifar10/MultipleNetworksCifar10.cpp
        Cifar10Database.hpp
        Cifar10Database.cpp)
    CaffeParserTest(MultipleNetworksCifar10 "${MultipleNetworksCifar10_SRC}")

    set(CaffeResNet-Armnn_sources
        CaffeResNet-Armnn/CaffeResNet-Armnn.cpp
        CaffePreprocessor.hpp
        CaffePreprocessor.cpp)
    CaffeParserTest(CaffeResNet-Armnn "${CaffeResNet-Armnn_sources}")

    set(CaffeVGG-Armnn_sources
        CaffeVGG-Armnn/CaffeVGG-Armnn.cpp
        CaffePreprocessor.hpp
        CaffePreprocessor.cpp)
    CaffeParserTest(CaffeVGG-Armnn "${CaffeVGG-Armnn_sources}")

    set(CaffeInception_BN-Armnn_sources
        CaffeInception_BN-Armnn/CaffeInception_BN-Armnn.cpp
        CaffePreprocessor.hpp
        CaffePreprocessor.cpp)
    CaffeParserTest(CaffeInception_BN-Armnn "${CaffeInception_BN-Armnn_sources}")

    set(CaffeYolo-Armnn_sources
        CaffeYolo-Armnn/CaffeYolo-Armnn.cpp
        YoloDatabase.hpp
        YoloDatabase.cpp
        YoloInferenceTest.hpp)
    CaffeParserTest(CaffeYolo-Armnn "${CaffeYolo-Armnn_sources}")
endif()

if(BUILD_TF_PARSER)
    macro(TfParserTest testName sources)
        add_executable_ex(${testName} ${sources})
        target_include_directories(${testName} PRIVATE ../src/armnnUtils)
        target_include_directories(${testName} PRIVATE ../src/backends)

        target_link_libraries(${testName} inferenceTest)
        target_link_libraries(${testName} armnnTfParser)
        target_link_libraries(${testName} armnn)
        target_link_libraries(${testName} ${CMAKE_THREAD_LIBS_INIT})
        addDllCopyCommands(${testName})
    endmacro()

    set(TfMnist-Armnn_sources
        TfMnist-Armnn/TfMnist-Armnn.cpp
        MnistDatabase.hpp
        MnistDatabase.cpp)
    TfParserTest(TfMnist-Armnn "${TfMnist-Armnn_sources}")

    set(TfCifar10-Armnn_sources
        TfCifar10-Armnn/TfCifar10-Armnn.cpp
        Cifar10Database.hpp
        Cifar10Database.cpp)
    TfParserTest(TfCifar10-Armnn "${TfCifar10-Armnn_sources}")

    set(TfMobileNet-Armnn_sources
        TfMobileNet-Armnn/TfMobileNet-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfParserTest(TfMobileNet-Armnn "${TfMobileNet-Armnn_sources}")

    set(TfInceptionV3-Armnn_sources
        TfInceptionV3-Armnn/TfInceptionV3-Armnn.cpp
        ImagePreprocessor.hpp
        ImagePreprocessor.cpp)
    TfParserTest(TfInceptionV3-Armnn "${TfInceptionV3-Armnn_sources}")

    set(TfResNext-Armnn_sources
        TfResNext_Quantized-Armnn/TfResNext_Quantized-Armnn.cpp
        CaffePreprocessor.hpp
        CaffePreprocessor.cpp)
    TfParserTest(TfResNext-Armnn "${TfResNext-Armnn_sources}")
endif()

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

if (BUILD_ARMNN_SERIALIZER OR BUILD_CAFFE_PARSER OR BUILD_TF_PARSER OR BUILD_TF_LITE_PARSER OR BUILD_ONNX_PARSER)
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
    if (BUILD_CAFFE_PARSER)
        target_link_libraries(ExecuteNetwork armnnCaffeParser)
    endif()
    if (BUILD_TF_PARSER)
        target_link_libraries(ExecuteNetwork armnnTfParser)
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
        if (BUILD_CAFFE_PARSER)
            target_link_libraries(${executorName} armnnCaffeParser)
        endif()
        if (BUILD_TF_PARSER)
            target_link_libraries(${executorName} armnnTfParser)
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

if(BUILD_ARMNN_QUANTIZER)
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
