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

# Include the additional cmake files in their own target so that they will appear nicely in IDEs
add_custom_target(AdditionalCMakeFiles SOURCES ${additional_cmake_files})

set(armnnUtils_sources)
list(APPEND armnnUtils_sources
    src/armnnUtils/GraphTopologicalSort.hpp
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

# ArmNN source files required for all build options
list(APPEND armnn_sources
    include/armnn/ArmNN.hpp
    include/armnn/Descriptors.hpp
    include/armnn/DescriptorsFwd.hpp
    include/armnn/IRuntime.hpp
    include/armnn/INetwork.hpp
    include/armnn/Tensor.hpp
    include/armnn/TensorFwd.hpp
    include/armnn/Types.hpp
    include/armnn/TypesUtils.hpp
    include/armnn/Utils.hpp
    include/armnn/LayerSupport.hpp
    include/armnn/Version.hpp
    src/armnn/backends/ClWorkloadFactory.hpp
    src/armnn/backends/ClWorkloadFactory.cpp
    src/armnn/backends/ClContextControl.hpp
    src/armnn/backends/ClContextControl.cpp
    src/armnn/backends/ClLayerSupport.cpp
    src/armnn/backends/ClLayerSupport.hpp
    src/armnn/backends/CpuTensorHandleFwd.hpp
    src/armnn/backends/CpuTensorHandle.hpp
    src/armnn/backends/CpuTensorHandle.cpp
    src/armnn/backends/RefWorkloadFactory.cpp
    src/armnn/backends/RefWorkloadFactory.hpp
    src/armnn/backends/RefLayerSupport.cpp
    src/armnn/backends/RefLayerSupport.hpp
    src/armnn/backends/MakeWorkloadHelper.hpp
    src/armnn/backends/NeonWorkloadFactory.cpp
    src/armnn/backends/NeonWorkloadFactory.hpp
    src/armnn/backends/NeonLayerSupport.cpp
    src/armnn/backends/NeonLayerSupport.hpp
    src/armnn/backends/Workload.hpp
    src/armnn/backends/WorkloadDataFwd.hpp
    src/armnn/backends/WorkloadData.hpp
    src/armnn/backends/WorkloadData.cpp
    src/armnn/backends/WorkloadFactory.hpp
    src/armnn/backends/WorkloadFactory.cpp
    src/armnn/backends/AclBaseMemoryManager.hpp
    src/armnn/backends/AclBaseMemoryManager.cpp
    src/armnn/backends/WorkloadInfo.hpp
    src/armnn/backends/MemCopyWorkload.cpp
    src/armnn/backends/MemCopyWorkload.hpp
    src/armnn/backends/RefWorkloads/Broadcast.hpp
    src/armnn/backends/RefWorkloads/Broadcast.cpp
    src/armnn/backends/RefWorkloads/RefMergerUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefConstantUint8Workload.hpp
    src/armnn/backends/RefWorkloads/Addition.hpp
    src/armnn/backends/RefWorkloads/ConvImpl.hpp
    src/armnn/backends/RefWorkloads/RefResizeBilinearUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefMultiplicationUint8Workload.hpp
    src/armnn/backends/RefWorkloads/FullyConnected.cpp
    src/armnn/backends/RefWorkloads/RefFullyConnectedFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefSoftmaxFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefBaseConstantWorkload.hpp
    src/armnn/backends/RefWorkloads/RefConvolution2dUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefSplitterUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefActivationFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefBatchNormalizationUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefResizeBilinearUint8Workload.hpp
    src/armnn/backends/RefWorkloads/Multiplication.cpp
    src/armnn/backends/RefWorkloads/RefMultiplicationUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefL2NormalizationFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/Multiplication.hpp
    src/armnn/backends/RefWorkloads/RefActivationUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefBaseConstantWorkload.cpp
    src/armnn/backends/RefWorkloads/RefAdditionUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefResizeBilinearFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefBatchNormalizationFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefPooling2dFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefWorkloadUtils.hpp
    src/armnn/backends/RefWorkloads/RefMergerUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefFullyConnectedFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/Softmax.hpp
    src/armnn/backends/RefWorkloads/RefMergerFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/Addition.cpp
    src/armnn/backends/RefWorkloads/RefFakeQuantizationFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/TensorBufferArrayView.hpp
    src/armnn/backends/RefWorkloads/ResizeBilinear.cpp
    src/armnn/backends/RefWorkloads/RefBatchNormalizationFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/Splitter.hpp
    src/armnn/backends/RefWorkloads/RefFullyConnectedUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefSoftmaxUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefReshapeFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefDepthwiseConvolution2dUint8Workload.hpp
    src/armnn/backends/RefWorkloads/FullyConnected.hpp
    src/armnn/backends/RefWorkloads/RefFloorFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefDepthwiseConvolution2dFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefPooling2dUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefFloorFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/ConvImpl.cpp
    src/armnn/backends/RefWorkloads/RefSoftmaxFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefSoftmaxUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefReshapeUint8Workload.hpp
    src/armnn/backends/RefWorkloads/Activation.cpp
    src/armnn/backends/RefWorkloads/RefResizeBilinearFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefAdditionUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefReshapeUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefMultiplicationFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefL2NormalizationFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefConvolution2dFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefBatchNormalizationUint8Workload.hpp
    src/armnn/backends/RefWorkloads/RefConvolution2dUint8Workload.cpp
    src/armnn/backends/RefWorkloads/ResizeBilinear.hpp
    src/armnn/backends/RefWorkloads/RefNormalizationFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefDepthwiseConvolution2dFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefSplitterFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefActivationUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefPooling2dUint8Workload.hpp
    src/armnn/backends/RefWorkloads/BatchNormImpl.hpp
    src/armnn/backends/RefWorkloads/RefSplitterUint8Workload.cpp
    src/armnn/backends/RefWorkloads/Activation.hpp
    src/armnn/backends/RefWorkloads/RefPooling2dFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/Merger.hpp
    src/armnn/backends/RefWorkloads/RefSplitterFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefConstantFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefActivationFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefMultiplicationFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefReshapeFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefNormalizationFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/Softmax.cpp
    src/armnn/backends/RefWorkloads/RefDepthwiseConvolution2dUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefConstantUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefConstantFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/Pooling2d.cpp
    src/armnn/backends/RefWorkloads/RefAdditionFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/RefConvolution2dFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefAdditionFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefMergerFloat32Workload.cpp
    src/armnn/backends/RefWorkloads/Pooling2d.hpp
    src/armnn/backends/RefWorkloads/RefFullyConnectedUint8Workload.cpp
    src/armnn/backends/RefWorkloads/RefFakeQuantizationFloat32Workload.hpp
    src/armnn/backends/RefWorkloads/RefPermuteWorkload.hpp
    src/armnn/backends/RefWorkloads/RefPermuteWorkload.cpp
    src/armnn/layers/LayerCloneBase.hpp
    src/armnn/layers/LayerWithParameters.hpp
    src/armnn/layers/ActivationLayer.hpp
    src/armnn/layers/ActivationLayer.cpp
    src/armnn/layers/AdditionLayer.hpp
    src/armnn/layers/AdditionLayer.cpp
    src/armnn/layers/BatchNormalizationLayer.hpp
    src/armnn/layers/BatchNormalizationLayer.cpp
    src/armnn/layers/ConstantLayer.hpp
    src/armnn/layers/ConstantLayer.cpp
    src/armnn/layers/Convolution2dLayer.hpp
    src/armnn/layers/Convolution2dLayer.cpp
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
    src/armnn/layers/PermuteLayer.hpp
    src/armnn/layers/PermuteLayer.cpp
    src/armnn/layers/Pooling2dLayer.hpp
    src/armnn/layers/Pooling2dLayer.cpp
    src/armnn/layers/ReshapeLayer.hpp
    src/armnn/layers/ReshapeLayer.cpp
    src/armnn/layers/ResizeBilinearLayer.hpp
    src/armnn/layers/ResizeBilinearLayer.cpp
    src/armnn/layers/SoftmaxLayer.hpp
    src/armnn/layers/SoftmaxLayer.cpp
    src/armnn/layers/SplitterLayer.hpp
    src/armnn/layers/SplitterLayer.cpp
    src/armnn/InternalTypes.hpp
    src/armnn/InternalTypes.cpp
    src/armnn/LayerFwd.hpp
    src/armnn/Layer.hpp
    src/armnn/Layer.cpp
    src/armnn/LayersFwd.hpp
    src/armnn/Runtime.hpp
    src/armnn/Runtime.cpp
    src/armnn/SerializeLayerParameters.cpp
    src/armnn/SerializeLayerParameters.hpp
    src/armnn/Descriptors.cpp
    src/armnn/LoadedNetwork.hpp
    src/armnn/LoadedNetwork.cpp
    src/armnn/Exceptions.cpp
    src/armnn/Graph.hpp
    src/armnn/Graph.cpp
    src/armnn/Network.hpp
    src/armnn/Network.cpp
    src/armnn/backends/OutputHandler.hpp
    src/armnn/backends/OutputHandler.cpp
    src/armnn/Profiling.cpp
    src/armnn/Tensor.cpp
    src/armnn/Utils.cpp
    src/armnn/LayerSupport.cpp
    src/armnn/LayerSupportCommon.hpp
    src/armnn/optimizations/All.hpp
    src/armnn/optimizations/MovePermuteUp.hpp
    src/armnn/optimizations/Optimization.hpp
    src/armnn/optimizations/OptimizeConsecutiveReshapes.hpp
    src/armnn/optimizations/OptimizeInversePermutes.hpp
    src/armnn/optimizations/PermuteAsReshape.hpp
    src/armnn/optimizations/SquashEqualSiblings.hpp
    src/armnn/Optimizer.hpp
    src/armnn/Optimizer.cpp
    )

if(ARMCOMPUTENEON)
    # Additionally include source files for ARM Compute NEON backend
    list(APPEND armnn_sources
        src/armnn/backends/NeonWorkloads/NeonActivationFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonActivationFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonActivationUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonActivationUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonAdditionFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonAdditionFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonBaseConstantWorkload.hpp
        src/armnn/backends/NeonWorkloads/NeonBaseMergerWorkload.hpp
        src/armnn/backends/NeonWorkloads/NeonBaseSplitterWorkload.hpp
        src/armnn/backends/NeonWorkloads/NeonBatchNormalizationFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonBatchNormalizationFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonConstantFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonConstantFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonConstantUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonConstantUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonConvolution2dBaseWorkload.cpp
        src/armnn/backends/NeonWorkloads/NeonConvolution2dBaseWorkload.hpp
        src/armnn/backends/NeonWorkloads/NeonConvolution2dFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonConvolution2dFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonConvolution2dUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonConvolution2dUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonDepthwiseConvolutionUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonFloorFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonFloorFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonFullyConnectedFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonFullyConnectedFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonL2NormalizationFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonL2NormalizationFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonMergerFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonMergerFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonMergerUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonMergerUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonMultiplicationFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonMultiplicationFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonNormalizationFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonNormalizationFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonPermuteWorkload.cpp
        src/armnn/backends/NeonWorkloads/NeonPermuteWorkload.hpp
        src/armnn/backends/NeonWorkloads/NeonPooling2dBaseWorkload.cpp
        src/armnn/backends/NeonWorkloads/NeonPooling2dBaseWorkload.hpp
        src/armnn/backends/NeonWorkloads/NeonPooling2dFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonPooling2dFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonPooling2dUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonPooling2dUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonReshapeFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonReshapeFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonReshapeUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonReshapeUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonSoftmaxFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonSoftmaxFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonSoftmaxUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonSoftmaxUint8Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonSplitterFloat32Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonSplitterFloat32Workload.hpp
        src/armnn/backends/NeonWorkloads/NeonSplitterUint8Workload.cpp
        src/armnn/backends/NeonWorkloads/NeonSplitterUint8Workload.hpp
        src/armnn/backends/NeonWorkloadUtils.cpp
        src/armnn/backends/NeonWorkloadUtils.hpp
        src/armnn/backends/NeonTensorHandle.hpp)
endif()
if(ARMCOMPUTECL)
    # Additionally include source files for ARM Compute OpenCL backend
    list(APPEND armnn_sources
        src/armnn/backends/ClWorkloads/ClActivationFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClActivationFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClActivationUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClActivationUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClAdditionFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClAdditionFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClBaseConstantWorkload.cpp
        src/armnn/backends/ClWorkloads/ClBaseConstantWorkload.hpp
        src/armnn/backends/ClWorkloads/ClBaseMergerWorkload.hpp
        src/armnn/backends/ClWorkloads/ClBatchNormalizationFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClBatchNormalizationFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClConstantFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClConstantFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClConstantUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClConstantUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClConvolution2dBaseWorkload.cpp
        src/armnn/backends/ClWorkloads/ClConvolution2dBaseWorkload.hpp
        src/armnn/backends/ClWorkloads/ClConvolution2dFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClConvolution2dFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClConvolution2dUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClConvolution2dUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClDepthwiseConvolutionHelper.hpp
        src/armnn/backends/ClWorkloads/ClFloorFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClFloorFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClFullyConnectedFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClFullyConnectedFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClL2NormalizationFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClL2NormalizationFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClMergerFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClMergerFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClMergerUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClMergerUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClMultiplicationFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClMultiplicationFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClNormalizationFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClNormalizationFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClPermuteWorkload.cpp
        src/armnn/backends/ClWorkloads/ClPermuteWorkload.hpp
        src/armnn/backends/ClWorkloads/ClPooling2dBaseWorkload.cpp
        src/armnn/backends/ClWorkloads/ClPooling2dBaseWorkload.hpp
        src/armnn/backends/ClWorkloads/ClPooling2dFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClPooling2dFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClPooling2dUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClPooling2dUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClReshapeFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClReshapeFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClReshapeUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClReshapeUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClResizeBilinearFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClResizeBilinearFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClSoftmaxFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClSoftmaxFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClSoftmaxUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClSoftmaxUint8Workload.hpp
        src/armnn/backends/ClWorkloads/ClSplitterFloat32Workload.cpp
        src/armnn/backends/ClWorkloads/ClSplitterFloat32Workload.hpp
        src/armnn/backends/ClWorkloads/ClSplitterUint8Workload.cpp
        src/armnn/backends/ClWorkloads/ClSplitterUint8Workload.hpp
        src/armnn/backends/ClWorkloadUtils.hpp
        src/armnn/backends/ClTensorHandle.hpp)
endif()
# Files shared by all ARM Compute backends
if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    list(APPEND armnn_sources
        src/armnn/backends/ArmComputeTensorUtils.hpp
        src/armnn/backends/ArmComputeTensorUtils.cpp
        src/armnn/backends/ArmComputeUtils.hpp)
endif()

# Files used for Streamline-based profiling backend
if(PROFILING_BACKEND_STREAMLINE)
    list(APPEND armnn_sources
        ${GATOR_ROOT}/annotate/streamline_annotate.h
        ${GATOR_ROOT}/annotate/streamline_annotate.c)
endif()

add_library_ex(armnn SHARED ${armnn_sources})
target_include_directories(armnn PRIVATE src/armnn)
target_include_directories(armnn PRIVATE src/armnnUtils)
target_link_libraries(armnn armnnUtils)
target_link_libraries(armnn ${CMAKE_DL_LIBS})

install(TARGETS armnn DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(BUILD_CAFFE_PARSER)
    install(TARGETS armnnCaffeParser DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()
if(BUILD_TF_PARSER)
    install(TARGETS armnnTfParser DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
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
        src/armnn/test/GraphTests.cpp
        src/armnn/test/OptimizerTests.cpp
        src/armnn/test/RuntimeTests.cpp
        src/armnn/test/CreateWorkload.hpp
        src/armnn/test/TensorTest.cpp
        src/armnn/test/TensorHelpers.hpp
        src/armnn/test/Network_test.cpp
        src/armnn/test/GraphUtils.hpp
        src/armnn/backends/test/IsLayerSupportedTest.cpp
        src/armnn/backends/test/IsLayerSupportedTestImpl.hpp
        src/armnn/backends/test/Reference.cpp
        src/armnn/backends/test/WorkloadDataValidation.cpp
        src/armnn/backends/test/TensorCopyUtils.hpp
        src/armnn/backends/test/TensorCopyUtils.cpp
        src/armnn/backends/test/LayerTests.hpp
        src/armnn/backends/test/LayerTests.cpp
        src/armnn/backends/test/Conv2dTestImpl.hpp
        src/armnn/backends/test/ActivationTestImpl.hpp
        src/armnn/backends/test/ActivationFixture.hpp
        src/armnn/backends/test/Pooling2dTestImpl.hpp
        src/armnn/backends/test/ReshapeTestImpl.hpp
        src/armnn/backends/test/PermuteTestImpl.hpp
        src/armnn/backends/test/FullyConnectedTestImpl.hpp
        src/armnn/backends/test/SplitterTestImpl.hpp
        src/armnn/backends/test/NormTestImpl.hpp
        src/armnn/backends/test/BatchNormTestImpl.hpp
        src/armnn/backends/test/WorkloadTestUtils.hpp
        src/armnn/backends/test/CreateWorkloadRef.cpp
        src/armnn/backends/test/QuantizeHelper.hpp)

    if(ARMCOMPUTENEON)
        list(APPEND unittest_sources
            src/armnn/backends/test/ArmComputeNeon.cpp
            src/armnn/backends/test/CreateWorkloadNeon.cpp
            src/armnn/test/CreateWorkloadClNeon.hpp)
    endif()

    if(ARMCOMPUTECL)
        list(APPEND unittest_sources
            src/armnn/backends/test/ArmComputeCl.cpp
            src/armnn/backends/test/CreateWorkloadCl.cpp
            src/armnn/test/CreateWorkloadClNeon.hpp)
    endif()

    if(ARMCOMPUTENEON OR ARMCOMPUTECL)
        list(APPEND unittest_sources
            src/armnn/backends/test/MemCopyTests.cpp)
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
            src/armnnTfParser/test/Multiplication.cpp
            src/armnnTfParser/test/MultiOutput.cpp
            src/armnnTfParser/test/PassThru.cpp
            src/armnnTfParser/test/Pooling.cpp
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

    if(BUILD_CAFFE_PARSER)
        list(APPEND unittest_sources
            src/armnnCaffeParser/test/TestAdd.cpp
            src/armnnCaffeParser/test/TestConcat.cpp
            src/armnnCaffeParser/test/TestDropout.cpp
            src/armnnCaffeParser/test/TestInputs.cpp
            src/armnnCaffeParser/test/TestMul.cpp
            src/armnnCaffeParser/test/TestPooling2d.cpp
            src/armnnCaffeParser/test/TestInPlace.cpp
            src/armnnCaffeParser/test/TestMultiInputsOutputs.cpp
            src/armnnCaffeParser/test/TestSplit.cpp
            )
    endif()

    add_executable_ex(UnitTests ${unittest_sources})
    target_include_directories(UnitTests PRIVATE src/armnn)
    target_include_directories(UnitTests PRIVATE src/armnnUtils)

    if(NOT HEAP_PROFILING AND VALGRIND_FOUND)
        # Valgrind works with gperftools version number <= 2.4
        target_compile_definitions(UnitTests PRIVATE "WITH_VALGRIND=1")
    endif()

    target_link_libraries(UnitTests armnn)
    target_link_libraries(UnitTests armnnUtils)
    target_link_libraries(UnitTests ${CMAKE_THREAD_LIBS_INIT})
    target_link_libraries(UnitTests ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

    if(BUILD_TF_PARSER)
        target_link_libraries(UnitTests armnnTfParser)
    endif()

    if(BUILD_CAFFE_PARSER)
        target_link_libraries(UnitTests armnnCaffeParser)
    endif()

    addDllCopyCommands(UnitTests)
endif()

