#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
cmake_minimum_required (VERSION 3.0.2) # 3.0.2 required for return() statement used in AddDllCopyCommands.cmake
project(armnn)

set(additional_cmake_files)
list(APPEND additional_cmake_files
    cmake/Utils.cmake
    cmake/GlobalConfig.cmake
    cmake/AddDllCopyCommands.cmake)

foreach(cmake_file ${additional_cmake_files})
    include(${cmake_file})
endforeach()

if (DYNAMIC_BACKEND_PATHS)
    # It's expected to have the format: DYNAMIC_BACKEND_PATHS="PATH_1:PATH_2...:PATH_N"
    add_definitions('-DDYNAMIC_BACKEND_PATHS="${DYNAMIC_BACKEND_PATHS}"')
endif()

include(GNUInstallDirs)

add_subdirectory(samples)
add_subdirectory(src/armnnTfLiteParser)
add_subdirectory(src/armnnSerializer)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Include the additional cmake files in their own target so that they will appear nicely in IDEs
add_custom_target(AdditionalCMakeFiles SOURCES ${additional_cmake_files})

set(armnnUtils_sources)
list(APPEND armnnUtils_sources
    src/armnnUtils/GraphTopologicalSort.hpp
    src/armnnUtils/Half.hpp
    src/armnnUtils/Logging.hpp
    src/armnnUtils/Logging.cpp
    src/armnnUtils/Permute.hpp
    src/armnnUtils/Permute.cpp
    src/armnnUtils/DataLayoutIndexed.cpp
    src/armnnUtils/DataLayoutIndexed.hpp
    src/armnnUtils/DotSerializer.cpp
    src/armnnUtils/DotSerializer.hpp
    src/armnnUtils/HeapProfiling.cpp
    src/armnnUtils/HeapProfiling.hpp
    src/armnnUtils/LeakChecking.cpp
    src/armnnUtils/LeakChecking.hpp
    src/armnnUtils/ModelAccuracyChecker.cpp
    src/armnnUtils/ModelAccuracyChecker.hpp
    src/armnnUtils/CsvReader.cpp
    src/armnnUtils/CsvReader.hpp
    src/armnnUtils/FloatingPointConverter.cpp
    src/armnnUtils/FloatingPointConverter.hpp
    src/armnnUtils/VerificationHelpers.hpp
    src/armnnUtils/VerificationHelpers.cpp
    src/armnnUtils/ParserHelper.hpp
    src/armnnUtils/ParserHelper.cpp
    src/armnnUtils/ParserPrototxtFixture.hpp
    src/armnnUtils/PrototxtConversions.hpp
    src/armnnUtils/PrototxtConversions.cpp
    src/armnnUtils/TensorIOUtils.hpp
    src/armnnUtils/TensorUtils.hpp
    src/armnnUtils/TensorUtils.cpp
    )

add_library_ex(armnnUtils STATIC ${armnnUtils_sources})
target_include_directories(armnnUtils PRIVATE src/backends)

if(BUILD_CAFFE_PARSER)
    # ArmNN Parser source files required for all build options
    set(armnn_caffe_parser_sources)
    list(APPEND armnn_caffe_parser_sources
        include/armnnCaffeParser/ICaffeParser.hpp
        src/armnnCaffeParser/RecordByRecordCaffeParser.hpp
        src/armnnCaffeParser/RecordByRecordCaffeParser.cpp
        src/armnnCaffeParser/CaffeParser.hpp
        src/armnnCaffeParser/CaffeParser.cpp
        ${CAFFE_GENERATED_SOURCES}/caffe/proto/caffe.pb.cc
        )
    # The generated Caffe protobuf .cc file is not warning clean and we can't fix them.
    if(COMPILER_IS_GNU_LIKE)
        set_source_files_properties(${CAFFE_GENERATED_SOURCES}/caffe/proto/caffe.pb.cc PROPERTIES COMPILE_FLAGS "-Wno-conversion -Wno-sign-conversion")
    endif()

    add_library_ex(armnnCaffeParser SHARED ${armnn_caffe_parser_sources})
    set_target_properties(armnnCaffeParser PROPERTIES COMPILE_FLAGS "${CAFFE_PARSER_ADDITIONAL_COMPILE_FLAGS}")

    target_include_directories(armnnCaffeParser PRIVATE src/armnnUtils)

    target_link_libraries(armnnCaffeParser ${Boost_LOG_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})

    target_link_libraries(armnnCaffeParser armnn)
    target_link_libraries(armnnCaffeParser ${PROTOBUF_LIBRARIES})

endif()

if(BUILD_ONNX_PARSER)
    set(armnn_onnx_parser_sources)
    list(APPEND armnn_onnx_parser_sources
        include/armnnOnnxParser/IOnnxParser.hpp
        src/armnnOnnxParser/OnnxParser.hpp
        src/armnnOnnxParser/OnnxParser.cpp
        ${ONNX_GENERATED_SOURCES}/onnx/onnx.pb.cc
        )
    # The generated onnx protobuf .cc files are not warning clean and we can't fix them.
    if(COMPILER_IS_GNU_LIKE)
        set_source_files_properties(${ONNX_GENERATED_SOURCES}/onnx/onnx.pb.cc PROPERTIES COMPILE_FLAGS "-Wno-conversion -Wno-sign-conversion")
    endif()

    add_library_ex(armnnOnnxParser SHARED ${armnn_onnx_parser_sources})

    target_include_directories(armnnOnnxParser PRIVATE src/armnnUtils)

    target_link_libraries(armnnOnnxParser armnn)

    # Protobuf
    target_link_libraries(armnnOnnxParser ${PROTOBUF_LIBRARIES})
endif()

if(BUILD_TF_PARSER)
    set(armnn_tf_parser_sources)
    list(APPEND armnn_tf_parser_sources
        include/armnnTfParser/ITfParser.hpp
        src/armnnTfParser/TfParser.hpp
        src/armnnTfParser/TfParser.cpp
        ${TF_PROTOBUFS}
        )
    # The generated tensorflow protobuf .cc files are not warning clean and we can't fix them.
    if(COMPILER_IS_GNU_LIKE)
        set_source_files_properties(${TF_PROTOBUFS} PROPERTIES COMPILE_FLAGS "-Wno-conversion -Wno-sign-conversion")
    endif()

    add_library_ex(armnnTfParser SHARED ${armnn_tf_parser_sources})

    target_include_directories(armnnTfParser PRIVATE src/armnnUtils)

    target_link_libraries(armnnTfParser armnn)

    # Protobuf (use the specific version tensorflow wants)
    target_link_libraries(armnnTfParser ${PROTOBUF_LIBRARIES})
endif()

if(BUILD_ARMNN_QUANTIZER AND ARMNNREF)
    if(NOT BUILD_ARMNN_SERIALIZER)
        message(ERROR, "In order to build the ArmNN Quantization Tool you must set BUILD_ARMNN_SERIALIZER = YES")
    endif()

    set(armnn_quantizer_sources)
    list(APPEND armnn_quantizer_sources
        src/armnnQuantizer/QuantizationDataSet.hpp
        src/armnnQuantizer/QuantizationDataSet.cpp
        src/armnnQuantizer/QuantizationInput.hpp
        src/armnnQuantizer/QuantizationInput.cpp
        src/armnnQuantizer/CommandLineProcessor.hpp
        src/armnnQuantizer/CommandLineProcessor.cpp
        )

    add_library_ex(armnnQuantizer SHARED ${armnn_quantizer_sources})

    target_include_directories(armnnQuantizer PRIVATE include/armnnDeserializer)
    target_include_directories(armnnQuantizer PRIVATE src/armnnUtils)
    target_include_directories(armnnQuantizer PRIVATE src/armnn)

    include_directories(SYSTEM "${FLATBUFFERS_INCLUDE_PATH}")
    set_target_properties(armnnQuantizer PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

    target_link_libraries(armnnQuantizer
            armnn
            ${Boost_SYSTEM_LIBRARY}
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_LOG_LIBRARY}
            ${Boost_THREAD_LIBRARY} )

    add_executable_ex(ArmnnQuantizer
        src/armnnQuantizer/ArmNNQuantizerMain.cpp)

    target_include_directories(ArmnnQuantizer PRIVATE include/armnnDeserializer)
    target_include_directories(ArmnnQuantizer PRIVATE src/armnn)

    target_link_libraries(ArmnnQuantizer
            ${Boost_SYSTEM_LIBRARY}
            ${Boost_PROGRAM_OPTIONS_LIBRARY}
            ${Boost_FILESYSTEM_LIBRARY}
            ${Boost_LOG_LIBRARY}
            ${Boost_THREAD_LIBRARY} )

    target_link_libraries(ArmnnQuantizer
            armnnQuantizer
            armnnSerializer
            armnn
            armnnUtils
            ${FLATBUFFERS_LIBRARY})

    if(Threads_FOUND AND (NOT ("${CMAKE_SYSTEM_NAME}" STREQUAL Android)))
        target_link_libraries(ArmnnQuantizer pthread)
    endif()

endif()


list(APPEND armnn_sources
    include/armnn/ArmNN.hpp
    include/armnn/BackendId.hpp
    include/armnn/Descriptors.hpp
    include/armnn/DescriptorsFwd.hpp
    include/armnn/Exceptions.hpp
    include/armnn/ILayerSupport.hpp
    include/armnn/ILayerVisitor.hpp
    include/armnn/INetwork.hpp
    include/armnn/IProfiler.hpp
    include/armnn/IRuntime.hpp
    include/armnn/LayerSupport.hpp
    include/armnn/LayerVisitorBase.hpp
    include/armnn/LstmParams.hpp
    include/armnn/MemorySources.hpp
    include/armnn/NetworkFwd.hpp
    include/armnn/Optional.hpp
    include/armnn/QuantizedLstmParams.hpp
    include/armnn/Tensor.hpp
    include/armnn/TensorFwd.hpp
    include/armnn/Types.hpp
    include/armnn/TypesUtils.hpp
    include/armnn/Utils.hpp
    include/armnn/Version.hpp
    src/armnn/layers/LayerCloneBase.hpp
    src/armnn/layers/LayerWithParameters.hpp
    src/armnn/layers/ActivationLayer.hpp
    src/armnn/layers/ActivationLayer.cpp
    src/armnn/layers/AdditionLayer.hpp
    src/armnn/layers/AdditionLayer.cpp
    src/armnn/layers/BatchNormalizationLayer.hpp
    src/armnn/layers/BatchNormalizationLayer.cpp
    src/armnn/layers/BatchToSpaceNdLayer.hpp
    src/armnn/layers/BatchToSpaceNdLayer.cpp
    src/armnn/layers/ConcatLayer.hpp
    src/armnn/layers/ConcatLayer.cpp
    src/armnn/layers/ConstantLayer.hpp
    src/armnn/layers/ConstantLayer.cpp
    src/armnn/layers/Convolution2dLayer.hpp
    src/armnn/layers/Convolution2dLayer.cpp
    src/armnn/layers/ConvertFp16ToFp32Layer.hpp
    src/armnn/layers/ConvertFp16ToFp32Layer.cpp
    src/armnn/layers/ConvertFp32ToFp16Layer.hpp
    src/armnn/layers/ConvertFp32ToFp16Layer.cpp
    src/armnn/layers/DebugLayer.hpp
    src/armnn/layers/DebugLayer.cpp
    src/armnn/layers/DepthwiseConvolution2dLayer.hpp
    src/armnn/layers/DepthwiseConvolution2dLayer.cpp
    src/armnn/layers/DequantizeLayer.hpp
    src/armnn/layers/DequantizeLayer.cpp
    src/armnn/layers/DetectionPostProcessLayer.hpp
    src/armnn/layers/DetectionPostProcessLayer.cpp
    src/armnn/layers/ElementwiseBaseLayer.hpp
    src/armnn/layers/ElementwiseBaseLayer.cpp
    src/armnn/layers/EqualLayer.hpp
    src/armnn/layers/EqualLayer.cpp
    src/armnn/layers/FakeQuantizationLayer.hpp
    src/armnn/layers/FakeQuantizationLayer.cpp
    src/armnn/layers/FloorLayer.hpp
    src/armnn/layers/FloorLayer.cpp
    src/armnn/layers/FullyConnectedLayer.hpp
    src/armnn/layers/FullyConnectedLayer.cpp
    src/armnn/layers/GatherLayer.cpp
    src/armnn/layers/GatherLayer.hpp
    src/armnn/layers/GreaterLayer.cpp
    src/armnn/layers/GreaterLayer.hpp
    src/armnn/layers/InputLayer.hpp
    src/armnn/layers/InputLayer.cpp
    src/armnn/layers/L2NormalizationLayer.hpp
    src/armnn/layers/L2NormalizationLayer.cpp
    src/armnn/layers/LstmLayer.cpp
    src/armnn/layers/LstmLayer.hpp
    src/armnn/layers/MaximumLayer.cpp
    src/armnn/layers/MaximumLayer.hpp
    src/armnn/layers/MeanLayer.hpp
    src/armnn/layers/MeanLayer.cpp
    src/armnn/layers/MemCopyLayer.hpp
    src/armnn/layers/MemCopyLayer.cpp
    src/armnn/layers/MemImportLayer.hpp
    src/armnn/layers/MemImportLayer.cpp
    src/armnn/layers/MergeLayer.hpp
    src/armnn/layers/MergeLayer.cpp
    src/armnn/layers/MinimumLayer.cpp
    src/armnn/layers/MinimumLayer.hpp
    src/armnn/layers/MultiplicationLayer.hpp
    src/armnn/layers/MultiplicationLayer.cpp
    src/armnn/layers/NormalizationLayer.hpp
    src/armnn/layers/NormalizationLayer.cpp
    src/armnn/layers/OutputLayer.hpp
    src/armnn/layers/OutputLayer.cpp
    src/armnn/layers/PadLayer.hpp
    src/armnn/layers/PadLayer.cpp
    src/armnn/layers/PermuteLayer.hpp
    src/armnn/layers/PermuteLayer.cpp
    src/armnn/layers/Pooling2dLayer.hpp
    src/armnn/layers/Pooling2dLayer.cpp
    src/armnn/layers/QuantizeLayer.cpp
    src/armnn/layers/QuantizeLayer.hpp
    src/armnn/layers/QuantizedLstmLayer.hpp
    src/armnn/layers/QuantizedLstmLayer.cpp
    src/armnn/layers/DivisionLayer.cpp
    src/armnn/layers/DivisionLayer.hpp
    src/armnn/layers/PreCompiledLayer.hpp
    src/armnn/layers/PreCompiledLayer.cpp
    src/armnn/layers/PreluLayer.hpp
    src/armnn/layers/PreluLayer.cpp
    src/armnn/layers/ReshapeLayer.hpp
    src/armnn/layers/ReshapeLayer.cpp
    src/armnn/layers/ResizeLayer.hpp
    src/armnn/layers/ResizeLayer.cpp
    src/armnn/layers/RsqrtLayer.cpp
    src/armnn/layers/RsqrtLayer.hpp
    src/armnn/layers/SoftmaxLayer.hpp
    src/armnn/layers/SoftmaxLayer.cpp
    src/armnn/layers/SpaceToBatchNdLayer.hpp
    src/armnn/layers/SpaceToBatchNdLayer.cpp
    src/armnn/layers/SpaceToDepthLayer.hpp
    src/armnn/layers/SpaceToDepthLayer.cpp
    src/armnn/layers/SplitterLayer.hpp
    src/armnn/layers/SplitterLayer.cpp
    src/armnn/layers/StackLayer.hpp
    src/armnn/layers/StackLayer.cpp
    src/armnn/layers/StridedSliceLayer.cpp
    src/armnn/layers/StridedSliceLayer.hpp
    src/armnn/layers/SubtractionLayer.cpp
    src/armnn/layers/SubtractionLayer.hpp
    src/armnn/layers/SwitchLayer.cpp
    src/armnn/layers/SwitchLayer.hpp
    src/armnn/layers/TransposeConvolution2dLayer.cpp
    src/armnn/layers/TransposeConvolution2dLayer.hpp
    src/armnn/BackendSettings.hpp
    src/armnn/BackendHelper.cpp
    src/armnn/CompatibleTypes.hpp
    src/armnn/Descriptors.cpp
    src/armnn/DeviceSpec.hpp
    src/armnn/DynamicQuantizationVisitor.cpp
    src/armnn/DynamicQuantizationVisitor.hpp
    src/armnn/Exceptions.cpp
    src/armnn/ExecutionFrame.cpp
    src/armnn/ExecutionFrame.hpp
    src/armnn/Graph.cpp
    src/armnn/Graph.hpp
    src/armnn/IGraphObservable.hpp
    src/armnn/Instrument.hpp
    src/armnn/InternalTypes.cpp
    src/armnn/InternalTypes.hpp
    src/armnn/ISubgraphViewConverter.hpp
    src/armnn/JsonPrinter.cpp
    src/armnn/JsonPrinter.hpp
    src/armnn/Layer.cpp
    src/armnn/LayerFwd.hpp
    src/armnn/Layer.hpp
    src/armnn/LayersFwd.hpp
    src/armnn/LayerSupportCommon.hpp
    src/armnn/LayerSupport.cpp
    src/armnn/LoadedNetwork.cpp
    src/armnn/LoadedNetwork.hpp
    src/armnn/Network.cpp
    src/armnn/Network.hpp
    src/armnn/NetworkQuantizationScheme.hpp
    src/armnn/NetworkQuantizer.cpp
    src/armnn/NetworkQuantizer.hpp
    src/armnn/NetworkQuantizerUtils.cpp
    src/armnn/NetworkQuantizerUtils.hpp
    src/armnn/NetworkUtils.cpp
    src/armnn/NetworkUtils.hpp
    src/armnn/Observable.cpp
    src/armnn/Observable.hpp
    src/armnn/Optimizer.cpp
    src/armnn/Optimizer.hpp
    src/armnn/OverrideInputRangeVisitor.cpp
    src/armnn/OverrideInputRangeVisitor.hpp
    src/armnn/Profiling.cpp
    src/armnn/ProfilingEvent.cpp
    src/armnn/ProfilingEvent.hpp
    src/armnn/Profiling.hpp
    src/armnn/QuantizerVisitor.cpp
    src/armnn/QuantizerVisitor.hpp
    src/armnn/Runtime.cpp
    src/armnn/Runtime.hpp
    src/armnn/RangeTracker.cpp
    src/armnn/RangeTracker.hpp
    src/armnn/ResolveType.hpp
    src/armnn/SerializeLayerParameters.cpp
    src/armnn/SerializeLayerParameters.hpp
    src/armnn/StaticRangeVisitor.cpp
    src/armnn/StaticRangeVisitor.hpp
    src/armnn/SubgraphView.cpp
    src/armnn/SubgraphView.hpp
    src/armnn/SubgraphViewSelector.cpp
    src/armnn/SubgraphViewSelector.hpp
    src/armnn/Tensor.cpp
    src/armnn/TypesUtils.cpp
    src/armnn/Utils.cpp
    src/armnn/WallClockTimer.cpp
    src/armnn/WallClockTimer.hpp
    src/armnn/optimizations/AddDebug.hpp
    src/armnn/optimizations/All.hpp
    src/armnn/optimizations/ConvertConstants.hpp
    src/armnn/optimizations/ConvertFp32NetworkToFp16.hpp
    src/armnn/optimizations/FoldPadIntoConvolution2d.hpp
    src/armnn/optimizations/MovePermuteUp.hpp
    src/armnn/optimizations/Optimization.hpp
    src/armnn/optimizations/OptimizeConsecutiveReshapes.hpp
    src/armnn/optimizations/OptimizeInverseConversions.hpp
    src/armnn/optimizations/OptimizeInversePermutes.hpp
    src/armnn/optimizations/PermuteAsReshape.hpp
    src/armnn/optimizations/SquashEqualSiblings.hpp
    src/profiling/CommandHandlerFunctor.cpp
    src/profiling/CommandHandlerFunctor.hpp
    src/profiling/CommandHandlerKey.cpp
    src/profiling/CommandHandlerKey.hpp
    src/profiling/CommandHandlerRegistry.cpp
    src/profiling/CommandHandlerRegistry.hpp
    src/profiling/EncodeVersion.hpp
    src/profiling/Packet.cpp
    src/profiling/Packet.hpp
    src/profiling/PacketVersionResolver.cpp
    src/profiling/PacketVersionResolver.hpp
    third-party/half/half.hpp
    )

# Files used for Streamline-based profiling backend
if(PROFILING_BACKEND_STREAMLINE)
    list(APPEND armnn_sources
        ${GATOR_ROOT}/annotate/streamline_annotate.h
        ${GATOR_ROOT}/annotate/streamline_annotate.c)
endif()

# the backends under src/backends extend the list of
# object libs armnn to include in the build
include(src/backends/backends.cmake)
foreach(lib ${armnnLibraries})
    message("Adding object library dependency to armnn: ${lib}")
    list(APPEND armnn_sources $<TARGET_OBJECTS:${lib}>)
endforeach()

add_library_ex(armnn SHARED ${armnn_sources})

target_include_directories(armnn PRIVATE src/armnn)
target_include_directories(armnn PRIVATE src/armnnUtils)
target_include_directories(armnn PRIVATE src/backends)

target_link_libraries(armnn armnnUtils)

target_link_libraries(armnn ${CMAKE_DL_LIBS})

install(TARGETS armnn
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(BUILD_CAFFE_PARSER)
    install(TARGETS armnnCaffeParser
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(BUILD_ONNX_PARSER)
    install(TARGETS armnnOnnxParser
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(BUILD_TF_PARSER)
    install(TARGETS armnnTfParser
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

target_link_libraries(armnn ${Boost_LOG_LIBRARY} ${Boost_THREAD_LIBRARY}
                            ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    target_link_libraries(armnn ${ARMCOMPUTE_LIBRARIES})
endif()
if(ARMCOMPUTECL)
    target_link_libraries(armnn ${OPENCL_LIBRARIES})
endif()

if(PROFILING_BACKEND_STREAMLINE)
    target_link_libraries(armnn pthread)
endif()

if(BUILD_UNIT_TESTS)
    set(unittest_sources)
    list(APPEND unittest_sources
        src/armnn/test/ConstTensorLayerVisitor.hpp
        src/armnn/test/ConstTensorLayerVisitor.cpp
        src/armnn/test/CreateWorkload.hpp
        src/armnn/test/CsvReaderTest.cpp
        src/armnn/test/EndToEndTest.cpp
        src/armnn/test/ExecutionFrameTest.cpp
        src/armnn/test/FloatingPointConverterTest.cpp
        src/armnn/test/FlowControl.cpp
        src/armnn/test/GraphTests.cpp
        src/armnn/test/GraphUtils.cpp
        src/armnn/test/GraphUtils.hpp
        src/armnn/test/InstrumentTests.cpp
        src/armnn/test/InferOutputTests.cpp
        src/armnn/test/InferOutputTests.hpp
        src/armnn/test/ModelAccuracyCheckerTest.cpp
        src/armnn/test/NetworkTests.cpp
        src/armnn/test/ObservableTest.cpp
        src/armnn/test/OptimizerTests.cpp
        src/armnn/test/OptionalTest.cpp
        src/armnn/test/ProfilerTests.cpp
        src/armnn/test/ProfilingEventTest.cpp
        src/armnn/test/SubgraphViewTests.cpp
        src/armnn/test/TensorHandleStrategyTest.cpp
        src/armnn/test/TensorHelpers.hpp
        src/armnn/test/TensorTest.cpp
        src/armnn/test/TestInputOutputLayerVisitor.cpp
        src/armnn/test/TestInputOutputLayerVisitor.hpp
        src/armnn/test/TestLayerVisitor.cpp
        src/armnn/test/TestLayerVisitor.hpp
        src/armnn/test/TestNameOnlyLayerVisitor.cpp
        src/armnn/test/TestNameOnlyLayerVisitor.hpp
        src/armnn/test/TestNameAndDescriptorLayerVisitor.hpp
        src/armnn/test/TestNameAndDescriptorLayerVisitor.cpp
        src/armnn/test/TestUtils.cpp
        src/armnn/test/TestUtils.hpp
        src/armnn/test/UnitTests.cpp
        src/armnn/test/UnitTests.hpp
        src/armnn/test/UtilsTests.cpp
        src/armnnUtils/test/PrototxtConversionsTest.cpp
        src/armnnUtils/test/ParserHelperTest.cpp
        src/profiling/test/ProfilingTests.cpp
        )

    if(ARMNNREF)
        list(APPEND unittest_sources
            src/armnn/test/DebugCallbackTest.cpp
            src/armnn/test/QuantizerTest.cpp
            src/armnn/test/RuntimeTests.cpp
            src/armnn/test/RuntimeTests.hpp
            )
    endif()

    if(BUILD_TF_PARSER AND ARMNNREF)
        list(APPEND unittest_sources
            src/armnnTfParser/test/Activations.cpp
            src/armnnTfParser/test/Addition.cpp
            src/armnnTfParser/test/AddN.cpp
            src/armnnTfParser/test/Assert.cpp
            src/armnnTfParser/test/BiasAdd.cpp
            src/armnnTfParser/test/BroadcastForAdd.cpp
            src/armnnTfParser/test/Convolution2d.cpp
            src/armnnTfParser/test/Concat.cpp
            src/armnnTfParser/test/ConcatOfConcats.cpp
            src/armnnTfParser/test/DepthwiseConvolution2d.cpp
            src/armnnTfParser/test/Equal.cpp
            src/armnnTfParser/test/ExpandDims.cpp
            src/armnnTfParser/test/FusedBatchNorm.cpp
            src/armnnTfParser/test/Gather.cpp
            src/armnnTfParser/test/Greater.cpp
            src/armnnTfParser/test/Identity.cpp
            src/armnnTfParser/test/LocalResponseNormalization.cpp
            src/armnnTfParser/test/Maximum.cpp
            src/armnnTfParser/test/MaximumForLeakyRelu.cpp
            src/armnnTfParser/test/Mean.cpp
            src/armnnTfParser/test/Minimum.cpp
            src/armnnTfParser/test/Multiplication.cpp
            src/armnnTfParser/test/Pad.cpp
            src/armnnTfParser/test/MultiOutput.cpp
            src/armnnTfParser/test/PassThru.cpp
            src/armnnTfParser/test/Pooling.cpp
            src/armnnTfParser/test/RealDiv.cpp
            src/armnnTfParser/test/Reshape.cpp
            src/armnnTfParser/test/ResizeBilinear.cpp
            src/armnnTfParser/test/Rsqrt.cpp
            src/armnnTfParser/test/Shape.cpp
            src/armnnTfParser/test/Softmax.cpp
            src/armnnTfParser/test/TestDependencies.cpp
            src/armnnTfParser/test/FullyConnected.cpp
            src/armnnTfParser/test/Constant.cpp
            src/armnnTfParser/test/TestMultiInputsOutputs.cpp
            src/armnnTfParser/test/Split.cpp
            src/armnnTfParser/test/Squeeze.cpp
            src/armnnTfParser/test/Sub.cpp
            )

    endif()

    if(BUILD_TF_LITE_PARSER AND ARMNNREF)
        enable_language(ASM)
        list(APPEND unittest_sources
             src/armnnTfLiteParser/test/Activations.cpp
             src/armnnTfLiteParser/test/Addition.cpp
             src/armnnTfLiteParser/test/AvgPool2D.cpp
             src/armnnTfLiteParser/test/BatchToSpaceND.cpp
             src/armnnTfLiteParser/test/Concatenation.cpp
             src/armnnTfLiteParser/test/Constant.cpp
             src/armnnTfLiteParser/test/Conv2D.cpp
             src/armnnTfLiteParser/test/DepthwiseConvolution2D.cpp
             src/armnnTfLiteParser/test/DetectionPostProcess.cpp
             src/armnnTfLiteParser/test/FullyConnected.cpp
             src/armnnTfLiteParser/test/L2Normalization.cpp
             src/armnnTfLiteParser/test/Maximum.cpp
             src/armnnTfLiteParser/test/MaxPool2D.cpp
             src/armnnTfLiteParser/test/Mean.cpp
             src/armnnTfLiteParser/test/Minimum.cpp
             src/armnnTfLiteParser/test/Multiplication.cpp
             src/armnnTfLiteParser/test/Pack.cpp
             src/armnnTfLiteParser/test/Pad.cpp
             src/armnnTfLiteParser/test/Reshape.cpp
             src/armnnTfLiteParser/test/ResizeBilinear.cpp
             src/armnnTfLiteParser/test/Softmax.cpp
             src/armnnTfLiteParser/test/SpaceToBatchND.cpp
             src/armnnTfLiteParser/test/Split.cpp
             src/armnnTfLiteParser/test/Squeeze.cpp
             src/armnnTfLiteParser/test/StridedSlice.cpp
             src/armnnTfLiteParser/test/Sub.cpp
             src/armnnTfLiteParser/test/TransposeConv.cpp
             src/armnnTfLiteParser/test/Unpack.cpp
             src/armnnTfLiteParser/test/LoadModel.cpp
             src/armnnTfLiteParser/test/GetBuffer.cpp
             src/armnnTfLiteParser/test/OutputShapeOfSqueeze.cpp
             src/armnnTfLiteParser/test/InputOutputTensorNames.cpp
             src/armnnTfLiteParser/test/GetTensorIds.cpp
             src/armnnTfLiteParser/test/GetSubgraphInputsOutputs.cpp
             src/armnnTfLiteParser/test/GetInputsOutputs.cpp
             src/armnnTfLiteParser/test/Schema.s
             )
        set_source_files_properties(src/armnnTfLiteParser/test/Schema.s PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")

    endif()

    if(BUILD_CAFFE_PARSER AND ARMNNREF)
        list(APPEND unittest_sources
            src/armnnCaffeParser/test/TestAdd.cpp
            src/armnnCaffeParser/test/TestConcat.cpp
            src/armnnCaffeParser/test/TestConvolution.cpp
            src/armnnCaffeParser/test/TestDropout.cpp
            src/armnnCaffeParser/test/TestInputs.cpp
            src/armnnCaffeParser/test/TestMul.cpp
            src/armnnCaffeParser/test/TestPooling2d.cpp
            src/armnnCaffeParser/test/TestInPlace.cpp
            src/armnnCaffeParser/test/TestMultiInputsOutputs.cpp
            src/armnnCaffeParser/test/TestSplit.cpp
            )
    endif()

    if(BUILD_ONNX_PARSER AND ARMNNREF)
        list(APPEND unittest_sources
            src/armnnOnnxParser/test/Constructor.cpp
            src/armnnOnnxParser/test/CreateNetwork.cpp
            src/armnnOnnxParser/test/ProtoxtFixture.cpp
            src/armnnOnnxParser/test/Const.cpp
            src/armnnOnnxParser/test/Pooling.cpp
            src/armnnOnnxParser/test/Reshape.cpp
            src/armnnOnnxParser/test/Relu.cpp
            src/armnnOnnxParser/test/Conv2D.cpp
            src/armnnOnnxParser/test/Addition.cpp
            src/armnnOnnxParser/test/FullyConnected.cpp
            src/armnnOnnxParser/test/GetInputsOutputs.cpp
            src/armnnOnnxParser/test/BatchNorm.cpp
            src/armnnOnnxParser/test/DepthConv.cpp
            )
    endif()

    if(BUILD_ARMNN_QUANTIZER AND ARMNNREF)
        list(APPEND unittest_sources
             src/armnnQuantizer/test/QuantizationDataSetTests.cpp
             )
    endif()

    if(BUILD_ARMNN_SERIALIZER AND ARMNNREF)
        enable_language(ASM)
        list(APPEND unittest_sources
            src/armnnSerializer/test/ActivationSerializationTests.cpp
            src/armnnSerializer/test/SerializerTests.cpp
            src/armnnDeserializer/test/DeserializeActivation.cpp
            src/armnnDeserializer/test/DeserializeAdd.cpp
            src/armnnDeserializer/test/DeserializeBatchToSpaceNd.cpp
            src/armnnDeserializer/test/DeserializeBatchNormalization.cpp
            src/armnnDeserializer/test/DeserializeConstant.cpp
            src/armnnDeserializer/test/DeserializeConvolution2d.cpp
            src/armnnDeserializer/test/DeserializeDivision.cpp
            src/armnnDeserializer/test/DeserializeEqual.cpp
            src/armnnDeserializer/test/DeserializeFloor.cpp
            src/armnnDeserializer/test/DeserializeFullyConnected.cpp
            src/armnnDeserializer/test/DeserializeGather.cpp
            src/armnnDeserializer/test/DeserializeGreater.cpp
            src/armnnDeserializer/test/DeserializeL2Normalization.cpp
            src/armnnDeserializer/test/DeserializeMean.cpp
            src/armnnDeserializer/test/DeserializeMultiplication.cpp
            src/armnnDeserializer/test/DeserializeNormalization.cpp
            src/armnnDeserializer/test/DeserializePad.cpp
            src/armnnDeserializer/test/DeserializePermute.cpp
            src/armnnDeserializer/test/DeserializePooling2d.cpp
            src/armnnDeserializer/test/DeserializeReshape.cpp
            src/armnnDeserializer/test/DeserializeResizeBilinear.cpp
            src/armnnDeserializer/test/DeserializeRsqrt.cpp
            src/armnnDeserializer/test/DeserializeSpaceToBatchNd.cpp
            src/armnnDeserializer/test/DeserializeStridedSlice.cpp
            src/armnnDeserializer/test/DeserializeSubtraction.cpp
            src/armnnDeserializer/test/ParserFlatbuffersSerializeFixture.hpp
            src/armnnDeserializer/test/SchemaSerialize.s
            )
        set_source_files_properties(src/armnnDeserializer/test/SchemaSerialize.s PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    endif()

    foreach(lib ${armnnUnitTestLibraries})
        message("Adding object library dependency to UnitTests: ${lib}")
        list(APPEND unittest_sources $<TARGET_OBJECTS:${lib}>)
    endforeach()

    add_executable(UnitTests ${unittest_sources})
    target_include_directories(UnitTests PRIVATE src/armnn)
    target_include_directories(UnitTests PRIVATE src/armnnUtils)
    target_include_directories(UnitTests PRIVATE src/backends)

    if(VALGRIND_FOUND)
        if(HEAP_PROFILING OR LEAK_CHECKING)
            message("Valgrind is disabled for heap profiling and leak checking builds.")
        else()
            # Valgrind works with gperftools version number <= 2.4
            target_compile_definitions(UnitTests PRIVATE "WITH_VALGRIND=1")
        endif()
    endif()

    target_link_libraries(UnitTests ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(UnitTests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

    target_link_libraries(UnitTests armnn)
    target_link_libraries(UnitTests armnnUtils)

    if(BUILD_TF_PARSER)
        target_link_libraries(UnitTests armnnTfParser)
    endif()

    if(BUILD_TF_LITE_PARSER)
        target_include_directories(UnitTests SYSTEM PRIVATE "${TF_LITE_SCHEMA_INCLUDE_PATH}")
        target_include_directories(UnitTests SYSTEM PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
        target_link_libraries(UnitTests armnnTfLiteParser)
    endif()

    if(BUILD_CAFFE_PARSER)
        target_link_libraries(UnitTests armnnCaffeParser)
    endif()

    if(BUILD_ARMNN_SERIALIZER AND ARMNNREF)
        target_include_directories(UnitTests SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src/armnnSerializer)
        target_include_directories(UnitTests SYSTEM PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
        target_link_libraries(UnitTests armnnSerializer)
    endif()

    if(BUILD_ARMNN_QUANTIZER AND ARMNNREF)
        target_include_directories(UnitTests SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src/armnnQuantizer)
        target_include_directories(UnitTests SYSTEM PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
        target_link_libraries(UnitTests armnnQuantizer armnnSerializer)
    endif()

    if(BUILD_ONNX_PARSER)
        target_link_libraries(UnitTests armnnOnnxParser)
    endif()

    addDllCopyCommands(UnitTests)
endif()

if (BUILD_ARMNN_SERIALIZER AND (BUILD_TF_PARSER OR BUILD_TF_LITE_PARSER OR BUILD_ONNX_PARSER OR BUILD_CAFFE_PARSER) AND ARMNNREF)
    set(ArmnnConverter_sources
        src/armnnConverter/ArmnnConverter.cpp)

    add_executable_ex(ArmnnConverter ${ArmnnConverter_sources})
    target_include_directories(ArmnnConverter PRIVATE src/armnn)
    target_include_directories(ArmnnConverter PRIVATE src/armnnUtils)

    if(BUILD_CAFFE_PARSER)
        target_link_libraries(ArmnnConverter armnnCaffeParser)
    endif()

    if(BUILD_ONNX_PARSER)
        target_link_libraries(ArmnnConverter armnnOnnxParser)
    endif()

    if(BUILD_TF_PARSER)
        target_link_libraries(ArmnnConverter armnnTfParser)
    endif()

    if(BUILD_TF_LITE_PARSER)
        target_link_libraries(ArmnnConverter armnnTfLiteParser)
    endif()

    target_link_libraries(ArmnnConverter armnnSerializer)

    target_link_libraries(ArmnnConverter armnn)
    target_link_libraries(ArmnnConverter ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(ArmnnConverter
        ${Boost_SYSTEM_LIBRARY}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_PROGRAM_OPTIONS_LIBRARY})
    addDllCopyCommands(ArmnnConverter)
endif()
