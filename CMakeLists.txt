#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#
cmake_minimum_required (VERSION 3.0.2) # 3.0.2 required for return() statement used in AddDllCopyCommands.cmake.
project(armnn)

set(additional_cmake_files)
list(APPEND additional_cmake_files
    cmake/Utils.cmake
    cmake/GlobalConfig.cmake
    cmake/AddDllCopyCommands.cmake)

foreach(cmake_file ${additional_cmake_files})
    include(${cmake_file})
endforeach()

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

add_subdirectory(samples)

# Include the additional cmake files in their own target so that they will appear nicely in IDEs
add_custom_target(AdditionalCMakeFiles SOURCES ${additional_cmake_files})

set(armnnUtils_sources)
list(APPEND armnnUtils_sources
    src/armnnUtils/GraphTopologicalSort.hpp
    src/armnnUtils/Half.hpp
    src/armnnUtils/Logging.hpp
    src/armnnUtils/Permute.hpp
    src/armnnUtils/Logging.cpp
    src/armnnUtils/Permute.cpp
    src/armnnUtils/DotSerializer.cpp
    src/armnnUtils/DotSerializer.hpp
    src/armnnUtils/HeapProfiling.cpp
    src/armnnUtils/HeapProfiling.hpp
    src/armnnUtils/LeakChecking.cpp
    src/armnnUtils/LeakChecking.hpp
    src/armnnUtils/CsvReader.cpp
    src/armnnUtils/CsvReader.hpp
    src/armnnUtils/FloatingPointConverter.cpp
    src/armnnUtils/FloatingPointConverter.hpp
    src/armnnUtils/VerificationHelpers.hpp
    src/armnnUtils/VerificationHelpers.cpp
    src/armnnUtils/ParserHelper.hpp
    src/armnnUtils/ParserHelper.cpp
    )
if(BUILD_TF_PARSER OR BUILD_CAFFE_PARSER)
    list(APPEND armnnUtils_sources
    src/armnnUtils/ParserPrototxtFixture.hpp
    )
endif()

add_library_ex(armnnUtils STATIC ${armnnUtils_sources})

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

if(BUILD_TF_LITE_PARSER)
    set(armnn_tf_lite_parser_sources)
    list(APPEND armnn_tf_lite_parser_sources
        include/armnnTfLiteParser/ITfLiteParser.hpp
        src/armnnTfLiteParser/TfLiteParser.hpp
        src/armnnTfLiteParser/TfLiteParser.cpp
     )

     add_library_ex(armnnTfLiteParser SHARED ${armnn_tf_lite_parser_sources})

     target_include_directories(armnnTfLiteParser PRIVATE src/armnnUtils)

     target_link_libraries(armnnTfLiteParser ${Boost_FILESYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
     target_link_libraries(armnnTfLiteParser armnn ${FLATBUFFERS_LIBRARY})
endif()

list(APPEND armnn_sources
    include/armnn/ArmNN.hpp
    include/armnn/Descriptors.hpp
    include/armnn/DescriptorsFwd.hpp
    include/armnn/IRuntime.hpp
    include/armnn/ILayerSupport.hpp
    include/armnn/INetwork.hpp
    include/armnn/Optional.hpp
    include/armnn/Tensor.hpp
    include/armnn/TensorFwd.hpp
    include/armnn/Types.hpp
    include/armnn/TypesUtils.hpp
    include/armnn/Utils.hpp
    include/armnn/LayerSupport.hpp
    include/armnn/Version.hpp
    src/armnn/layers/LayerCloneBase.hpp
    src/armnn/layers/LayerWithParameters.hpp
    src/armnn/layers/ActivationLayer.hpp
    src/armnn/layers/ActivationLayer.cpp
    src/armnn/layers/AdditionLayer.hpp
    src/armnn/layers/AdditionLayer.cpp
    src/armnn/layers/ArithmeticBaseLayer.hpp
    src/armnn/layers/ArithmeticBaseLayer.cpp
    src/armnn/layers/BatchNormalizationLayer.hpp
    src/armnn/layers/BatchNormalizationLayer.cpp
    src/armnn/layers/ConstantLayer.hpp
    src/armnn/layers/ConstantLayer.cpp
    src/armnn/layers/Convolution2dLayer.hpp
    src/armnn/layers/Convolution2dLayer.cpp
    src/armnn/layers/ConvertFp16ToFp32Layer.hpp
    src/armnn/layers/ConvertFp16ToFp32Layer.cpp
    src/armnn/layers/ConvertFp32ToFp16Layer.hpp
    src/armnn/layers/ConvertFp32ToFp16Layer.cpp
    src/armnn/layers/DepthwiseConvolution2dLayer.hpp
    src/armnn/layers/DepthwiseConvolution2dLayer.cpp
    src/armnn/layers/FakeQuantizationLayer.hpp
    src/armnn/layers/FakeQuantizationLayer.cpp
    src/armnn/layers/FloorLayer.hpp
    src/armnn/layers/FloorLayer.cpp
    src/armnn/layers/FullyConnectedLayer.hpp
    src/armnn/layers/FullyConnectedLayer.cpp
    src/armnn/layers/InputLayer.hpp
    src/armnn/layers/InputLayer.cpp
    src/armnn/layers/L2NormalizationLayer.hpp
    src/armnn/layers/L2NormalizationLayer.cpp
    src/armnn/layers/LstmLayer.cpp
    src/armnn/layers/LstmLayer.hpp
    src/armnn/layers/MeanLayer.hpp
    src/armnn/layers/MeanLayer.cpp
    src/armnn/layers/MemCopyLayer.hpp
    src/armnn/layers/MemCopyLayer.cpp
    src/armnn/layers/MergerLayer.hpp
    src/armnn/layers/MergerLayer.cpp
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
    src/armnn/layers/DivisionLayer.cpp
    src/armnn/layers/DivisionLayer.hpp
    src/armnn/layers/ReshapeLayer.hpp
    src/armnn/layers/ReshapeLayer.cpp
    src/armnn/layers/ResizeBilinearLayer.hpp
    src/armnn/layers/ResizeBilinearLayer.cpp
    src/armnn/layers/SoftmaxLayer.hpp
    src/armnn/layers/SoftmaxLayer.cpp
    src/armnn/layers/SplitterLayer.hpp
    src/armnn/layers/SplitterLayer.cpp
    src/armnn/layers/SubtractionLayer.cpp
    src/armnn/layers/SubtractionLayer.hpp
    src/armnn/InternalTypes.hpp
    src/armnn/InternalTypes.cpp
    src/armnn/JsonPrinter.hpp
    src/armnn/JsonPrinter.cpp
    src/armnn/LayerFwd.hpp
    src/armnn/Layer.hpp
    src/armnn/Layer.cpp
    src/armnn/LayersFwd.hpp
    src/armnn/Runtime.hpp
    src/armnn/Runtime.cpp
    src/armnn/SerializeLayerParameters.cpp
    src/armnn/SerializeLayerParameters.hpp
    src/armnn/Descriptors.cpp
    src/armnn/DeviceSpec.hpp
    src/armnn/LoadedNetwork.hpp
    src/armnn/LoadedNetwork.cpp
    src/armnn/Exceptions.cpp
    src/armnn/Graph.hpp
    src/armnn/Graph.cpp
    src/armnn/Network.hpp
    src/armnn/Network.cpp
    src/armnn/NetworkUtils.hpp
    src/armnn/ProfilingEvent.cpp
    src/armnn/ProfilingEvent.hpp
    src/armnn/Profiling.cpp
    src/armnn/Instrument.hpp
    src/armnn/WallClockTimer.hpp
    src/armnn/WallClockTimer.cpp
    src/armnn/Tensor.cpp
    src/armnn/Utils.cpp
    src/armnn/LayerSupport.cpp
    src/armnn/LayerSupportCommon.hpp
    src/armnn/optimizations/All.hpp
    src/armnn/optimizations/ConvertConstants.hpp
    src/armnn/optimizations/MovePermuteUp.hpp
    src/armnn/optimizations/Optimization.hpp
    src/armnn/optimizations/OptimizeConsecutiveReshapes.hpp
    src/armnn/optimizations/OptimizeInversePermutes.hpp
    src/armnn/optimizations/PermuteAsReshape.hpp
    src/armnn/optimizations/SquashEqualSiblings.hpp
    src/armnn/optimizations/OptimizeInverseConversions.hpp
    src/armnn/optimizations/ConvertFp32NetworkToFp16.hpp
    src/armnn/Optimizer.hpp
    src/armnn/Optimizer.cpp
    third-party/half/half.hpp
    src/armnn/IGraphObservable.hpp
    src/armnn/Observable.hpp
    src/armnn/Observable.cpp
    )

if(ARMCOMPUTENEON)
    # Additionally include source files for ARM Compute NEON backend
    list(APPEND armnn_sources
        src/armnn/NeonInterceptorScheduler.hpp
        src/armnn/NeonInterceptorScheduler.cpp
        src/armnn/NeonTimer.hpp
        src/armnn/NeonTimer.cpp)
endif()
if(ARMCOMPUTECL)
    # Additionally include source files for ARM Compute OpenCL backend
    list(APPEND armnn_sources
        src/armnn/OpenClTimer.cpp
        src/armnn/OpenClTimer.hpp)
endif()
# Files shared by all ARM Compute backends
if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    list(APPEND armnn_sources
        src/armnn/memory/IMemoryPool.hpp
        src/armnn/memory/BlobMemoryPool.cpp
        src/armnn/memory/BlobMemoryPool.hpp
        src/armnn/memory/BlobLifetimeManager.cpp
        src/armnn/memory/BlobLifetimeManager.hpp
        src/armnn/memory/PoolManager.cpp
        src/armnn/memory/PoolManager.hpp
        src/armnn/memory/BaseMemoryManager.hpp
        src/armnn/memory/BaseMemoryManager.cpp
        src/armnn/memory/OffsetMemoryPool.cpp
        src/armnn/memory/OffsetMemoryPool.hpp
        src/armnn/memory/OffsetLifetimeManager.cpp
        src/armnn/memory/OffsetLifetimeManager.hpp)
endif()

# Files used for Streamline-based profiling backend
if(PROFILING_BACKEND_STREAMLINE)
    list(APPEND armnn_sources
        ${GATOR_ROOT}/annotate/streamline_annotate.h
        ${GATOR_ROOT}/annotate/streamline_annotate.c)
endif()

add_library_ex(armnn SHARED ${armnn_sources})

# the backends under src/backends extend the list of
# static libs armnn to link against
list(APPEND armnnLibraries armnnUtils)
include(src/backends/backends.cmake)

target_include_directories(armnn PRIVATE src)
target_include_directories(armnn PRIVATE src/armnn)
target_include_directories(armnn PRIVATE src/armnnUtils)
foreach(lib ${armnnLibraries})
    target_link_libraries(armnn ${lib})
    message("Adding library dependency to armnn: ${lib}")
endforeach()
target_link_libraries(armnn ${CMAKE_DL_LIBS})

install(TARGETS armnn DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_CAFFE_PARSER)
    install(TARGETS armnnCaffeParser DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(BUILD_ONNX_PARSER)
    install(TARGETS armnnOnnxParser DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(BUILD_TF_PARSER)
    install(TARGETS armnnTfParser DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(BUILD_TF_LITE_PARSER)
    install(TARGETS armnnTfLiteParser DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)

target_link_libraries(armnn ${Boost_LOG_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_SYSTEM_LIBRARY})

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
        src/armnn/test/UnitTests.cpp
        src/armnn/test/UnitTests.hpp
        src/armnn/test/EndToEndTest.cpp
        src/armnn/test/UtilsTests.cpp
        src/armnn/test/JsonPrinterTests.cpp
        src/armnn/test/GraphTests.cpp
        src/armnn/test/OptimizerTests.cpp
        src/armnn/test/ProfilerTests.cpp
        src/armnn/test/RuntimeTests.cpp
        src/armnn/test/CreateWorkload.hpp
        src/armnn/test/TensorTest.cpp
        src/armnn/test/TensorHelpers.hpp
        src/armnn/test/CsvReaderTest.cpp
        src/armnn/test/NetworkTests.cpp
        src/armnn/test/FloatingPointConverterTest.cpp
        src/armnn/test/ProfilingEventTest.cpp
        src/armnn/test/GraphUtils.hpp
        src/armnn/test/InstrumentTests.cpp
        src/armnn/test/ObservableTest.cpp
        src/armnn/test/OptionalTest.cpp
        src/backends/test/IsLayerSupportedTest.cpp
        src/backends/test/IsLayerSupportedTestImpl.hpp
        src/backends/test/Reference.cpp
        src/backends/test/WorkloadDataValidation.cpp
        src/backends/test/TensorCopyUtils.hpp
        src/backends/test/TensorCopyUtils.cpp
        src/backends/test/LayerTests.hpp
        src/backends/test/LayerTests.cpp
        src/backends/test/LayerReleaseConstantDataTest.cpp
        src/backends/test/Conv2dTestImpl.hpp
        src/backends/test/ActivationTestImpl.hpp
        src/backends/test/ActivationFixture.hpp
        src/backends/test/Pooling2dTestImpl.hpp
        src/backends/test/ReshapeTestImpl.hpp
        src/backends/test/PermuteTestImpl.hpp
        src/backends/test/FullyConnectedTestImpl.hpp
        src/backends/test/SplitterTestImpl.hpp
        src/backends/test/NormTestImpl.hpp
        src/backends/test/BatchNormTestImpl.hpp
        src/backends/test/WorkloadTestUtils.hpp
        src/backends/test/CreateWorkloadRef.cpp
        src/backends/test/QuantizeHelper.hpp)

    if(ARMCOMPUTENEON)
        list(APPEND unittest_sources
            src/backends/test/ArmComputeNeon.cpp
            src/backends/test/CreateWorkloadNeon.cpp
            src/armnn/test/CreateWorkloadClNeon.hpp
            src/armnn/test/NeonTimerTest.cpp)
    endif()

    if(ARMCOMPUTECL)
        list(APPEND unittest_sources
            src/backends/test/ArmComputeCl.cpp
            src/backends/test/ClContextControlFixture.hpp
            src/backends/test/CreateWorkloadCl.cpp
            src/armnn/test/CreateWorkloadClNeon.hpp
            src/armnn/test/OpenClTimerTest.cpp
            src/armnn/test/FP16SupportTest.cpp)
    endif()

    if(ARMCOMPUTENEON OR ARMCOMPUTECL)
        list(APPEND unittest_sources
            src/backends/test/MemCopyTests.cpp)
    endif()

    if(BUILD_TF_PARSER)
        list(APPEND unittest_sources
            src/armnnTfParser/test/Activations.cpp
            src/armnnTfParser/test/Addition.cpp
            src/armnnTfParser/test/BiasAdd.cpp
            src/armnnTfParser/test/BroadcastForAdd.cpp
            src/armnnTfParser/test/Convolution2d.cpp
            src/armnnTfParser/test/Concat.cpp
            src/armnnTfParser/test/ConcatOfConcats.cpp
            src/armnnTfParser/test/DepthwiseConvolution2d.cpp
            src/armnnTfParser/test/FusedBatchNorm.cpp
            src/armnnTfParser/test/Identity.cpp
            src/armnnTfParser/test/LocalResponseNormalization.cpp
            src/armnnTfParser/test/MaximumForLeakyRelu.cpp
            src/armnnTfParser/test/Multiplication.cpp
            src/armnnTfParser/test/MultiOutput.cpp
            src/armnnTfParser/test/PassThru.cpp
            src/armnnTfParser/test/Pooling.cpp
            src/armnnTfParser/test/RealDiv.cpp
            src/armnnTfParser/test/Reshape.cpp
            src/armnnTfParser/test/ResizeBilinear.cpp
            src/armnnTfParser/test/Shape.cpp
            src/armnnTfParser/test/Softmax.cpp
            src/armnnTfParser/test/TestDependencies.cpp
            src/armnnTfParser/test/FullyConnected.cpp
            src/armnnTfParser/test/Constant.cpp
            src/armnnTfParser/test/TestMultiInputsOutputs.cpp
            src/armnnTfParser/test/Squeeze.cpp)
    endif()

    if(BUILD_TF_LITE_PARSER)
        list(APPEND unittest_sources
             src/armnnTfLiteParser/test/ParserFlatbuffersFixture.hpp
             src/armnnTfLiteParser/test/Activations.cpp
             src/armnnTfLiteParser/test/AvgPool2D.cpp
             src/armnnTfLiteParser/test/Concatenation.cpp
             src/armnnTfLiteParser/test/Conv2D.cpp
             src/armnnTfLiteParser/test/DepthwiseConvolution2D.cpp
             src/armnnTfLiteParser/test/Reshape.cpp
             src/armnnTfLiteParser/test/Softmax.cpp
             src/armnnTfLiteParser/test/Squeeze.cpp
             src/armnnTfLiteParser/test/LoadModel.cpp
             src/armnnTfLiteParser/test/GetBuffer.cpp
             src/armnnTfLiteParser/test/OutputShapeOfSqueeze.cpp
             src/armnnTfLiteParser/test/InputOutputTensorNames.cpp
             src/armnnTfLiteParser/test/GetTensorIds.cpp
             src/armnnTfLiteParser/test/GetSubgraphInputsOutputs.cpp
             src/armnnTfLiteParser/test/GetInputsOutputs.cpp
             )
    endif()

    if(BUILD_CAFFE_PARSER)
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

    if(BUILD_ONNX_PARSER)
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

    add_executable_ex(UnitTests ${unittest_sources})
    target_include_directories(UnitTests PRIVATE src)
    target_include_directories(UnitTests PRIVATE src/armnn)
    target_include_directories(UnitTests PRIVATE src/armnnUtils)

    if(VALGRIND_FOUND)
        if(HEAP_PROFILING OR LEAK_CHECKING)
            message("Valgrind is disabled for heap profiling and leak checking builds.")
        else()
            # Valgrind works with gperftools version number <= 2.4
            target_compile_definitions(UnitTests PRIVATE "WITH_VALGRIND=1")
        endif()
    endif()

    target_link_libraries(UnitTests armnn)
    target_link_libraries(UnitTests armnnUtils)
    target_link_libraries(UnitTests ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(UnitTests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY})

    if(BUILD_TF_PARSER)
        target_link_libraries(UnitTests armnnTfParser)
    endif()

    if(BUILD_CAFFE_PARSER)
        target_link_libraries(UnitTests armnnCaffeParser)
    endif()

    if(BUILD_TF_LITE_PARSER)
        target_link_libraries(UnitTests armnnTfLiteParser)
    endif()

    if(BUILD_ONNX_PARSER)
        target_link_libraries(UnitTests armnnOnnxParser)
    endif()

    addDllCopyCommands(UnitTests)
endif()
