#
# Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
# Copyright © 2020 NXP
# SPDX-License-Identifier: MIT
#
cmake_minimum_required (VERSION 3.7.0)
project(armnn)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(additional_cmake_files)
list(APPEND additional_cmake_files
    cmake/ArmnnVersion.cmake
    cmake/ClassicDelegateVersion.cmake
    cmake/OpaqueDelegateVersion.cmake
    cmake/ParserVersion.cmake
    cmake/Utils.cmake
    cmake/GlobalConfig.cmake
    cmake/AddDllCopyCommands.cmake
    cmake/ArmnnTestUtilsVersion.cmake)

foreach(cmake_file ${additional_cmake_files})
    include(${cmake_file})
endforeach()

cmake_policy(SET CMP0057 NEW)

# If the root of the tensorflow source tree has not been specified
# and we are trying to build the tensorflow lite parser make
# our best guess as to where it will be i.e. one directory above
# the one this CMakeLists.txt file lives in, in a subdirectory
# called tensorflow
if (BUILD_TF_LITE_PARSER AND "${TENSORFLOW_ROOT}" STREQUAL "")
  set(TENSORFLOW_ROOT "${PROJECT_SOURCE_DIR}/../tensorflow" CACHE INTERNAL "")
  message(STATUS "Set TENSORFLOW_ROOT: ${TENSORFLOW_ROOT}")
else()
  message(STATUS "TENSORFLOW_ROOT: ${TENSORFLOW_ROOT}")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/delegate/cmake/Modules/")

set(as_subproject Armnn)

macro(find_package)
    if(NOT ${ARGV0} IN_LIST as_subproject)
        _find_package(${ARGV})
    endif()
endmacro()

if (DYNAMIC_BACKEND_PATHS)
    # It's expected to have the format: DYNAMIC_BACKEND_PATHS="PATH_1:PATH_2...:PATH_N"
    add_definitions('-DDYNAMIC_BACKEND_PATHS="${DYNAMIC_BACKEND_PATHS}"')
endif()

include(GNUInstallDirs)

add_subdirectory(profiling/common/src)
add_subdirectory(profiling/client/src)

add_subdirectory(samples)
add_subdirectory(src/armnnTfLiteParser)
add_subdirectory(src/armnnSerializer)
add_subdirectory(src/armnnDeserializer)
add_subdirectory(src/armnnTestUtils)

if (BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Include the additional cmake files in their own target so that they will appear nicely in IDEs
add_custom_target(AdditionalCMakeFiles SOURCES ${additional_cmake_files})

set(armnnUtils_sources)
list(APPEND armnnUtils_sources
    include/armnnUtils/CompatibleTypes.hpp
    include/armnnUtils/DataLayoutIndexed.hpp
    include/armnnUtils/Permute.hpp
    include/armnnUtils/Filesystem.hpp
    include/armnnUtils/FloatingPointComparison.hpp
    include/armnnUtils/FloatingPointConverter.hpp
    include/armnnUtils/QuantizeHelper.hpp
    include/armnnUtils/TContainer.hpp
    include/armnnUtils/TensorUtils.hpp
    include/armnnUtils/Transpose.hpp
    src/armnnUtils/BFloat16.hpp
    src/armnnUtils/CompatibleTypes.cpp
    src/armnnUtils/Filesystem.cpp
    src/armnnUtils/GraphTopologicalSort.hpp
    src/armnnUtils/Half.hpp
    src/armnnUtils/Permute.cpp
    src/armnnUtils/DataLayoutIndexed.cpp
    src/armnnUtils/DotSerializer.cpp
    src/armnnUtils/DotSerializer.hpp
    src/armnnUtils/HeapProfiling.cpp
    src/armnnUtils/HeapProfiling.hpp
    src/armnnUtils/LeakChecking.cpp
    src/armnnUtils/LeakChecking.hpp
    src/armnnUtils/ModelAccuracyChecker.cpp
    src/armnnUtils/ModelAccuracyChecker.hpp
    src/armnnUtils/FloatingPointConverter.cpp
    src/armnnUtils/VerificationHelpers.hpp
    src/armnnUtils/VerificationHelpers.cpp
    src/armnnUtils/ParserHelper.hpp
    src/armnnUtils/ParserHelper.cpp
    src/armnnUtils/ParserPrototxtFixture.hpp
    src/armnnUtils/ProfilingOptionsConverter.hpp
    src/armnnUtils/ProfilingOptionsConverter.cpp
    src/armnnUtils/PrototxtConversions.hpp
    src/armnnUtils/PrototxtConversions.cpp
    src/armnnUtils/TensorIOUtils.hpp
    src/armnnUtils/TensorUtils.cpp
    src/armnnUtils/Transpose.cpp
    )

add_library_ex(armnnUtils STATIC ${armnnUtils_sources})

target_include_directories(armnnUtils PRIVATE src/backends)

if(BUILD_ONNX_PARSER)
    set(armnn_onnx_parser_sources)
    list(APPEND armnn_onnx_parser_sources
        include/armnnOnnxParser/IOnnxParser.hpp
        include/armnnOnnxParser/Version.hpp
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
    set_target_properties(armnnOnnxParser PROPERTIES VERSION ${ONNX_PARSER_LIB_VERSION} SOVERSION ${ONNX_PARSER_LIB_SOVERSION})
endif()

list(APPEND armnn_sources
    include/armnn/ArmNN.hpp
    include/armnn/BackendHelper.hpp
    include/armnn/BackendId.hpp
    include/armnn/BackendOptions.hpp
    include/armnn/BackendRegistry.hpp
    include/armnn/Conversion.hpp
    include/armnn/Deprecated.hpp
    include/armnn/Descriptors.hpp
    include/armnn/DescriptorsFwd.hpp
    include/armnn/Exceptions.hpp
    include/armnn/backends/ILayerSupport.hpp
    include/armnn/backends/ICustomAllocator.hpp
    include/armnn/IAsyncExecutionCallback.hpp
    include/armnn/INetwork.hpp
    include/armnn/IProfiler.hpp
    include/armnn/IRuntime.hpp
    include/armnn/IStrategy.hpp
    include/armnn/IWorkingMemHandle.hpp
    include/armnn/Logging.hpp
    include/armnn/LstmParams.hpp
    include/armnn/MemorySources.hpp
    include/armnn/NetworkFwd.hpp
    include/armnn/Optional.hpp
    include/armnn/QuantizedLstmParams.hpp
    include/armnn/StrategyBase.hpp
    include/armnn/Tensor.hpp
    include/armnn/TensorFwd.hpp
    include/armnn/Threadpool.hpp
    include/armnn/Types.hpp
    include/armnn/TypesUtils.hpp
    include/armnn/Utils.hpp
    include/armnn/Version.hpp
    include/armnn/utility/Assert.hpp
    include/armnn/utility/IgnoreUnused.hpp
    include/armnn/utility/NumericCast.hpp
    include/armnn/utility/PolymorphicDowncast.hpp
    include/armnn/utility/StringUtils.hpp
    include/armnn/utility/TransformIterator.hpp
    include/armnn/backends/SubgraphView.hpp
    src/armnn/layers/LayerCloneBase.hpp
    src/armnn/layers/LayerWithParameters.hpp
    src/armnn/layers/ActivationLayer.hpp
    src/armnn/layers/ActivationLayer.cpp
    src/armnn/layers/AdditionLayer.hpp
    src/armnn/layers/AdditionLayer.cpp
    src/armnn/layers/ArgMinMaxLayer.hpp
    src/armnn/layers/ArgMinMaxLayer.cpp
    src/armnn/layers/BatchMatMulLayer.hpp
    src/armnn/layers/BatchMatMulLayer.cpp
    src/armnn/layers/BatchNormalizationLayer.hpp
    src/armnn/layers/BatchNormalizationLayer.cpp
    src/armnn/layers/BatchToSpaceNdLayer.hpp
    src/armnn/layers/BatchToSpaceNdLayer.cpp
    src/armnn/layers/CastLayer.hpp
    src/armnn/layers/CastLayer.cpp
    src/armnn/layers/ChannelShuffleLayer.hpp
    src/armnn/layers/ChannelShuffleLayer.cpp
    src/armnn/layers/ComparisonLayer.hpp
    src/armnn/layers/ComparisonLayer.cpp
    src/armnn/layers/ConcatLayer.hpp
    src/armnn/layers/ConcatLayer.cpp
    src/armnn/layers/ConstantLayer.hpp
    src/armnn/layers/ConstantLayer.cpp
    src/armnn/layers/Convolution2dLayer.hpp
    src/armnn/layers/Convolution2dLayer.cpp
    src/armnn/layers/Convolution3dLayer.hpp
    src/armnn/layers/Convolution3dLayer.cpp
    src/armnn/layers/ConvertFp16ToFp32Layer.hpp
    src/armnn/layers/ConvertFp16ToFp32Layer.cpp
    src/armnn/layers/ConvertFp32ToFp16Layer.hpp
    src/armnn/layers/ConvertFp32ToFp16Layer.cpp
    src/armnn/layers/DebugLayer.hpp
    src/armnn/layers/DebugLayer.cpp
    src/armnn/layers/DepthToSpaceLayer.hpp
    src/armnn/layers/DepthToSpaceLayer.cpp
    src/armnn/layers/DepthwiseConvolution2dLayer.hpp
    src/armnn/layers/DepthwiseConvolution2dLayer.cpp
    src/armnn/layers/DequantizeLayer.hpp
    src/armnn/layers/DequantizeLayer.cpp
    src/armnn/layers/DetectionPostProcessLayer.hpp
    src/armnn/layers/DetectionPostProcessLayer.cpp
    src/armnn/layers/ElementwiseBaseLayer.hpp
    src/armnn/layers/ElementwiseBaseLayer.cpp
    src/armnn/layers/ElementwiseBinaryLayer.hpp
    src/armnn/layers/ElementwiseBinaryLayer.cpp
    src/armnn/layers/ElementwiseUnaryLayer.hpp
    src/armnn/layers/ElementwiseUnaryLayer.cpp
    src/armnn/layers/FakeQuantizationLayer.hpp
    src/armnn/layers/FakeQuantizationLayer.cpp
    src/armnn/layers/FillLayer.hpp
    src/armnn/layers/FillLayer.cpp
    src/armnn/layers/FloorLayer.hpp
    src/armnn/layers/FloorLayer.cpp
    src/armnn/layers/FullyConnectedLayer.hpp
    src/armnn/layers/FullyConnectedLayer.cpp
    src/armnn/layers/GatherLayer.cpp
    src/armnn/layers/GatherLayer.hpp
    src/armnn/layers/GatherNdLayer.cpp
    src/armnn/layers/GatherNdLayer.hpp
    src/armnn/layers/InputLayer.hpp
    src/armnn/layers/InputLayer.cpp
    src/armnn/layers/InstanceNormalizationLayer.hpp
    src/armnn/layers/InstanceNormalizationLayer.cpp
    src/armnn/layers/L2NormalizationLayer.hpp
    src/armnn/layers/L2NormalizationLayer.cpp
    src/armnn/layers/LogicalBinaryLayer.hpp
    src/armnn/layers/LogicalBinaryLayer.cpp
    src/armnn/layers/LogSoftmaxLayer.hpp
    src/armnn/layers/LogSoftmaxLayer.cpp
    src/armnn/layers/LstmLayer.cpp
    src/armnn/layers/LstmLayer.hpp
    src/armnn/layers/MapLayer.cpp
    src/armnn/layers/MapLayer.hpp
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
    src/armnn/layers/Pooling3dLayer.hpp
    src/armnn/layers/Pooling3dLayer.cpp
    src/armnn/layers/QuantizeLayer.cpp
    src/armnn/layers/QuantizeLayer.hpp
    src/armnn/layers/QLstmLayer.hpp
    src/armnn/layers/QLstmLayer.cpp
    src/armnn/layers/QuantizedLstmLayer.hpp
    src/armnn/layers/QuantizedLstmLayer.cpp
    src/armnn/layers/DivisionLayer.cpp
    src/armnn/layers/DivisionLayer.hpp
    src/armnn/layers/PreCompiledLayer.hpp
    src/armnn/layers/PreCompiledLayer.cpp
    src/armnn/layers/PreluLayer.hpp
    src/armnn/layers/PreluLayer.cpp
    src/armnn/layers/RankLayer.hpp
    src/armnn/layers/RankLayer.cpp
    src/armnn/layers/ReduceLayer.hpp
    src/armnn/layers/ReduceLayer.cpp
    src/armnn/layers/ReshapeLayer.hpp
    src/armnn/layers/ReshapeLayer.cpp
    src/armnn/layers/ResizeLayer.hpp
    src/armnn/layers/ResizeLayer.cpp
    src/armnn/layers/ShapeLayer.cpp
    src/armnn/layers/ShapeLayer.hpp
    src/armnn/layers/SliceLayer.cpp
    src/armnn/layers/SliceLayer.hpp
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
    src/armnn/layers/StandInLayer.cpp
    src/armnn/layers/StandInLayer.hpp
    src/armnn/layers/StridedSliceLayer.cpp
    src/armnn/layers/StridedSliceLayer.hpp
    src/armnn/layers/SubtractionLayer.cpp
    src/armnn/layers/SubtractionLayer.hpp
    src/armnn/layers/SwitchLayer.cpp
    src/armnn/layers/SwitchLayer.hpp
    src/armnn/layers/TransposeConvolution2dLayer.cpp
    src/armnn/layers/TransposeConvolution2dLayer.hpp
    src/armnn/layers/TransposeLayer.hpp
    src/armnn/layers/TransposeLayer.cpp
    src/armnn/layers/UnidirectionalSequenceLstmLayer.cpp
    src/armnn/layers/UnidirectionalSequenceLstmLayer.hpp
    src/armnn/layers/UnmapLayer.cpp
    src/armnn/layers/UnmapLayer.hpp
    src/armnn/ArmNNProfilingServiceInitialiser.cpp
    src/armnn/ArmNNProfilingServiceInitialiser.hpp
    src/armnn/AsyncExecutionCallback.cpp
    src/armnn/AsyncExecutionCallback.hpp
    src/armnn/BackendRegistry.cpp
    src/armnn/BackendSettings.hpp
    src/armnn/BackendHelper.cpp
    src/armnn/Descriptors.cpp
    src/armnn/DeviceSpec.hpp
    src/armnn/DllExport.hpp
    src/armnn/Exceptions.cpp
    src/armnn/ExecutionData.hpp
    src/armnn/ExecutionFrame.cpp
    src/armnn/ExecutionFrame.hpp
    src/armnn/Graph.cpp
    src/armnn/Graph.hpp
    src/armnn/IGraphObservable.hpp
    src/armnn/ILayerSupport.cpp
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
    src/armnn/LoadedNetwork.cpp
    src/armnn/LoadedNetwork.hpp
    src/armnn/Logging.cpp
    src/armnn/Network.cpp
    src/armnn/Network.hpp
    src/armnn/NetworkUtils.cpp
    src/armnn/NetworkUtils.hpp
    src/armnn/Observable.cpp
    src/armnn/Observable.hpp
    src/armnn/Optimizer.cpp
    src/armnn/Optimizer.hpp
    src/armnn/OutputHandler.cpp
    src/armnn/OutputHandler.hpp
    src/armnn/Profiling.cpp
    src/armnn/ProfilingEvent.cpp
    src/armnn/ProfilingDetails.hpp
    src/armnn/ProfilingEvent.hpp
    src/armnn/Profiling.hpp
    src/armnn/Runtime.cpp
    src/armnn/Runtime.hpp
    src/armnn/RangeTracker.cpp
    src/armnn/RangeTracker.hpp
    src/armnn/ResolveType.hpp
    src/armnn/SerializeLayerParameters.cpp
    src/armnn/SerializeLayerParameters.hpp
    src/armnn/SubgraphView.cpp
    src/armnn/SubgraphViewSelector.cpp
    src/armnn/SubgraphViewSelector.hpp
    src/armnn/Tensor.cpp
    src/armnn/Threadpool.cpp
    src/armnn/TypesUtils.cpp
    src/armnn/Utils.cpp
    src/armnn/WallClockTimer.cpp
    src/armnn/WallClockTimer.hpp
    src/armnn/WorkingMemDescriptor.hpp
    src/armnn/WorkingMemHandle.cpp
    src/armnn/WorkingMemHandle.hpp
    src/armnn/optimizations/AddBroadcastReshapeLayer.hpp
    src/armnn/optimizations/AddDebug.hpp
    src/armnn/optimizations/All.hpp
    src/armnn/optimizations/ConvertConstants.hpp
    src/armnn/optimizations/ConvertFp32NetworkToFp16.hpp
    src/armnn/optimizations/FoldPadIntoLayer2d.hpp
    src/armnn/optimizations/MovePermuteUp.hpp
    src/armnn/optimizations/MoveTransposeUp.hpp
    src/armnn/optimizations/Optimization.hpp
    src/armnn/optimizations/OptimizeConsecutiveReshapes.hpp
    src/armnn/optimizations/OptimizeInverseConversions.hpp
    src/armnn/optimizations/OptimizeInversePermutes.hpp
    src/armnn/optimizations/PermuteAndBatchToSpaceAsDepthToSpace.hpp
    src/armnn/optimizations/PermuteAsReshape.hpp
    src/armnn/optimizations/SquashEqualSiblings.hpp
    third-party/cxxopts/cxxopts.hpp
    third-party/ghc/filesystem.hpp
    third-party/half/half.hpp
    third-party/mapbox/optional.hpp
    third-party/mapbox/recursive_wrapper.hpp
    third-party/mapbox/variant.hpp
    third-party/mapbox/variant_cast.hpp
    third-party/mapbox/variant_io.hpp
    third-party/mapbox/variant_visitor.hpp
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
    message(STATUS "Adding object library dependency to armnn: ${lib}")
    list(APPEND armnn_sources $<TARGET_OBJECTS:${lib}>)
endforeach()

# The delegate needs to be placed after armnnLibraries has been fully populated. The armnn_delegate_jni library
# requires a static armnn build.
# Explanation:
# Because backends are added as object libraries they won't be linked to armnn when building armnn statically.
# A target that uses a static armnn library has to link to the object libraries in the variable armnnLibraries
# manually to include all symbols from backends.
if (BUILD_CLASSIC_DELEGATE OR BUILD_OPAQUE_DELEGATE)
    set(ARMNN_SUB_PROJECT ON)
    set(ARMNN_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_subdirectory(delegate)
endif()
if (BUILD_CLASSIC_DELEGATE)
    add_definitions(-DARMNN_TFLITE_DELEGATE)
endif()
if (BUILD_OPAQUE_DELEGATE)
    add_definitions(-DARMNN_TFLITE_OPAQUE_DELEGATE)
endif()

if(BUILD_BARE_METAL OR EXECUTE_NETWORK_STATIC)
    add_library_ex(armnn STATIC ${armnn_sources})
else()
    if (BUILD_SHARED_LIBS)
        add_library_ex(armnn SHARED ${armnn_sources})
    else()
        add_library(armnn ${armnn_sources})
    endif()
endif()

target_compile_definitions(armnn PRIVATE "ARMNN_COMPILING_DLL")

# Generate a map file for all build modes
set_property(TARGET armnn APPEND_STRING PROPERTY
        LINK_FLAGS " -Wl,-Map=libarmnnMapFile.map")
message(STATUS "Linker will generate mapfile ")

target_include_directories(armnn
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        src/armnn
        src/armnnUtils
        src/backends
        src/profiling
        profiling/common/include
        profiling/client/include
)

# Link fmt third-party library
add_subdirectory(third-party/fmt)
target_link_libraries(armnn PUBLIC fmt)

target_link_libraries(armnn PUBLIC armnnUtils)
target_link_libraries(armnn PUBLIC pipeCommon)
target_link_libraries(armnn PUBLIC pipeClient)

if(NOT BUILD_BARE_METAL AND NOT EXECUTE_NETWORK_STATIC)
    target_link_libraries(armnn PUBLIC ${CMAKE_DL_LIBS})
endif()

if ("${CMAKE_SYSTEM_NAME}" STREQUAL Windows)
    target_link_libraries(armnn PUBLIC Ws2_32.lib)
endif()


if(BUILD_ONNX_PARSER)
    install(TARGETS armnnOnnxParser
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if(ARMCOMPUTENEON OR ARMCOMPUTECL)
    target_link_libraries(armnn PUBLIC ${ARMCOMPUTE_LIBRARIES})
endif()

# Flatbuffers is not included in armnn if the armnnSerializer is not present
# Required to target link both flatbuffers and tosa_serialization_lib to armnn
if(ARMNNTOSACOMMON)
    if (FLATBUFFERS_LIBRARY)
        target_link_libraries(armnn PUBLIC ${FLATBUFFERS_LIBRARY})
    else()
        # Use PATH_SUFFIXES to help find separate libs for debug/release on Windows builds
        find_library(FLATBUFFERS_LIBRARY_DEBUG NAMES flatbuffers
                     HINTS ${FLATBUFFERS_BUILD_DIR}
                     PATH_SUFFIXES "Debug")
        find_library(FLATBUFFERS_LIBRARY_RELEASE NAMES flatbuffers
                     HINTS ${FLATBUFFERS_BUILD_DIR}
                     PATH_SUFFIXES "Release")
        target_link_libraries(armnn PUBLIC
                     debug ${FLATBUFFERS_LIBRARY_DEBUG}
                     optimized ${FLATBUFFERS_LIBRARY_RELEASE})
    endif()

    if (TOSA_SERIALIZATION_LIB)
        target_link_libraries(armnn PUBLIC -Wl,--whole-archive ${TOSA_SERIALIZATION_LIB} -Wl,--no-whole-archive)
    else()
        find_library(TOSA_SERIALIZATION_LIB
                     NAMES libtosa_serialization_lib.a tosa_serialization_lib
                     HINTS ${TOSA_SERIALIZATION_LIB_ROOT}/lib /usr/local/lib /usr/lib)

        message(STATUS "TOSA serialization library set to ${TOSA_SERIALIZATION_LIB}")

        target_link_libraries(armnn PUBLIC -Wl,--whole-archive ${TOSA_SERIALIZATION_LIB} -Wl,--no-whole-archive)
    endif()
endif()

if (ARMNNTOSAREF)
    if (TOSA_REFERENCE_MODEL_LIB)
        target_link_libraries(armnn PUBLIC -Wl,--whole-archive ${TOSA_REFERENCE_MODEL_LIB} -Wl,--no-whole-archive)
    else()
        find_library(TOSA_REFERENCE_MODEL_LIB
            NAMES libtosa_reference_model_lib.a tosa_reference_model_lib
            HINTS ${TOSA_REFERENCE_MODEL_ROOT}/lib /usr/local/lib /usr/lib)
        message(STATUS "TOSA Reference Model set to ${TOSA_REFERENCE_MODEL_LIB}")

        target_link_libraries(armnn PUBLIC -Wl,--whole-archive ${TOSA_REFERENCE_MODEL_LIB} -Wl,--no-whole-archive)
    endif()
endif()

if(PROFILING_BACKEND_STREAMLINE AND (NOT ("${CMAKE_SYSTEM_NAME}" STREQUAL Android)))
    target_link_libraries(armnn PUBLIC ${CMAKE_THREAD_LIBS_INIT})
endif()

set_target_properties(armnn PROPERTIES VERSION ${GENERIC_LIB_VERSION} SOVERSION ${GENERIC_LIB_SOVERSION})

if(BUILD_UNIT_TESTS)
    set(unittest_sources)
    list(APPEND unittest_sources
        src/armnn/test/CloneTests.cpp
        src/armnn/test/ConstTensorLayerVisitor.hpp
        src/armnn/test/ConstTensorLayerVisitor.cpp
        src/armnn/test/EndToEndTest.cpp
        src/armnn/test/ExecutionFrameTest.cpp
        src/armnn/test/FloatingPointConverterTest.cpp
        src/armnn/test/FlowControl.cpp
        src/armnn/test/GraphTests.cpp
        src/armnn/test/InstrumentTests.cpp
        src/armnn/test/InferOutputTests.cpp
        src/armnn/test/InferOutputTests.hpp
        src/armnn/test/ModelAccuracyCheckerTest.cpp
        src/armnn/test/NetworkTests.cpp
        src/armnn/test/ObservableTest.cpp
        src/armnn/test/OptimizerTests.cpp
        src/armnn/test/optimizations/AddBroadcastReshapeLayerTests.cpp
        src/armnn/test/optimizations/ConvertConstDequantisationLayersToConstLayersTest.cpp
        src/armnn/test/optimizations/ConvertConstPermuteLayersToConstLayersTest.cpp
        src/armnn/test/optimizations/ConvertConstantsFloatToHalfTests.cpp
        src/armnn/test/optimizations/ConvertConstantsHalfToFloatTests.cpp
        src/armnn/test/optimizations/FoldPadIntoQuantizedAveragePooling2DTests.cpp
        src/armnn/test/optimizations/FoldPadTests.cpp
        src/armnn/test/optimizations/Fp32NetworkToFp16ConverterTests.cpp
        src/armnn/test/optimizations/FuseActivationTests.cpp
        src/armnn/test/optimizations/InsertDebugLayerTests.cpp
        src/armnn/test/optimizations/MovePermuteUpTests.cpp
        src/armnn/test/optimizations/MoveTransposeUpTests.cpp
        src/armnn/test/optimizations/OptimizeConsecutiveReshapesTests.cpp
        src/armnn/test/optimizations/OptimizeInverseConversionsTests.cpp
        src/armnn/test/optimizations/OptimizeInversePermutesTests.cpp
        src/armnn/test/optimizations/PermuteAndBatchToSpaceAsDepthToSpaceTests.cpp
        src/armnn/test/optimizations/PermuteAsReshapeTests.cpp
        src/armnn/test/optimizations/ReduceMultipleAxesTests.cpp
        src/armnn/test/optimizations/SquashEqualSiblingsTests.cpp
        src/armnn/test/optimizations/TransposeAsReshapeTests.cpp
        src/armnn/test/OptionalTest.cpp
        src/armnn/test/ProfilerTests.cpp
        src/armnn/test/ProfilingEventTest.cpp
        src/armnn/test/ShapeInferenceTests.cpp
        src/armnn/test/SubgraphViewTests.cpp
        src/armnn/test/TensorHandleStrategyTest.cpp
        src/armnn/test/TensorTest.cpp
        src/armnn/test/TestInputOutputLayerVisitor.cpp
        src/armnn/test/TestInputOutputLayerVisitor.hpp
        src/armnn/test/TestLayerVisitor.cpp
        src/armnn/test/TestLayerVisitor.hpp
        src/armnn/test/TestNameOnlyLayerVisitor.cpp
        src/armnn/test/TestNameOnlyLayerVisitor.hpp
        src/armnn/test/TestNameAndDescriptorLayerVisitor.hpp
        src/armnn/test/TestNameAndDescriptorLayerVisitor.cpp
        src/armnn/test/UtilityTests.cpp
        src/armnn/test/UtilsTests.cpp
        src/armnnUtils/test/FloatingPointComparisonTest.cpp
        src/armnnUtils/test/ParserHelperTest.cpp
        src/armnnUtils/test/PrototxtConversionsTest.cpp
        src/armnnUtils/test/QuantizeHelperTest.cpp
        src/armnnUtils/test/TensorUtilsTest.cpp
        src/armnnUtils/test/TransformIteratorTest.cpp
        src/profiling/test/BufferTests.cpp
        src/profiling/test/FileOnlyProfilingDecoratorTests.cpp
        src/profiling/test/PrintPacketHeaderHandler.cpp
        src/profiling/test/PrintPacketHeaderHandler.hpp
        src/profiling/test/ProfilingConnectionDumpToFileDecoratorTests.cpp
        src/profiling/test/ProfilingGuidTest.cpp
        src/profiling/test/ProfilingMocks.hpp
        src/profiling/test/ProfilingTests.cpp
        src/profiling/test/ProfilingTests.hpp
        src/profiling/test/ProfilingTestUtils.cpp
        src/profiling/test/ProfilingTestUtils.hpp
        src/profiling/test/RequestCountersPacketHandler.cpp
        src/profiling/test/RequestCountersPacketHandler.hpp
        src/profiling/test/SendCounterPacketTests.cpp
        src/profiling/test/SendCounterPacketTests.hpp
        src/profiling/test/TestTimelinePacketHandler.cpp
        src/profiling/test/TestTimelinePacketHandler.hpp
        src/profiling/test/TimelineModel.cpp
        src/profiling/test/TimelineModel.hpp
        src/profiling/test/SendTimelinePacketTests.cpp
        src/profiling/test/TimelinePacketTests.cpp
        src/profiling/test/TimelineUtilityMethodsTests.cpp
        third-party/doctest/doctest.h
        )

    if(ARMNNREF)
        list(APPEND unittest_sources
            src/armnn/test/optimizations/FuseBatchNormTests.cpp
            src/armnn/test/DebugCallbackTest.cpp
            src/armnn/test/RuntimeTests.cpp
            src/armnn/test/RuntimeTests.hpp
            )
    endif()

    if(BUILD_TF_LITE_PARSER AND ARMNNREF)
        list(APPEND unittest_sources
             src/armnnTfLiteParser/test/Activations.cpp
             src/armnnTfLiteParser/test/Addition.cpp
             src/armnnTfLiteParser/test/ArgMinMax.cpp
             src/armnnTfLiteParser/test/AvgPool2D.cpp
             src/armnnTfLiteParser/test/BatchMatMul.cpp
             src/armnnTfLiteParser/test/BatchToSpaceND.cpp
             src/armnnTfLiteParser/test/Cast.cpp
             src/armnnTfLiteParser/test/Comparison.cpp
             src/armnnTfLiteParser/test/Concatenation.cpp
             src/armnnTfLiteParser/test/Constant.cpp
             src/armnnTfLiteParser/test/Conv2D.cpp
             src/armnnTfLiteParser/test/Conv3D.cpp
             src/armnnTfLiteParser/test/DepthwiseConvolution2D.cpp
             src/armnnTfLiteParser/test/DepthToSpace.cpp
             src/armnnTfLiteParser/test/Dequantize.cpp
             src/armnnTfLiteParser/test/DetectionPostProcess.cpp
             src/armnnTfLiteParser/test/Div.cpp
             src/armnnTfLiteParser/test/ElementWiseUnary.cpp
             src/armnnTfLiteParser/test/ExpandDims.cpp
             src/armnnTfLiteParser/test/FloorDiv.cpp
             src/armnnTfLiteParser/test/FullyConnected.cpp
             src/armnnTfLiteParser/test/Gather.cpp
             src/armnnTfLiteParser/test/GatherNd.cpp
             src/armnnTfLiteParser/test/L2Normalization.cpp
             src/armnnTfLiteParser/test/LeakyRelu.cpp
             src/armnnTfLiteParser/test/LoadScopeDynamicTensor.cpp
             src/armnnTfLiteParser/test/LocalResponseNormalization.cpp
             src/armnnTfLiteParser/test/Maximum.cpp
             src/armnnTfLiteParser/test/MaxPool2D.cpp
             src/armnnTfLiteParser/test/Mean.cpp
             src/armnnTfLiteParser/test/Minimum.cpp
             src/armnnTfLiteParser/test/MirrorPad.cpp
             src/armnnTfLiteParser/test/Multiplication.cpp
             src/armnnTfLiteParser/test/Pack.cpp
             src/armnnTfLiteParser/test/Pad.cpp
             src/armnnTfLiteParser/test/PadV2.cpp
             src/armnnTfLiteParser/test/Prelu.cpp
             src/armnnTfLiteParser/test/Reduce.cpp
             src/armnnTfLiteParser/test/Reshape.cpp
             src/armnnTfLiteParser/test/ReshapeDynamic.cpp
             src/armnnTfLiteParser/test/ResizeBilinear.cpp
             src/armnnTfLiteParser/test/ResizeNearestNeighbor.cpp
             src/armnnTfLiteParser/test/Quantize.cpp
             src/armnnTfLiteParser/test/Softmax.cpp
             src/armnnTfLiteParser/test/SpaceToBatchND.cpp
             src/armnnTfLiteParser/test/Shape.cpp
             src/armnnTfLiteParser/test/Slice.cpp
             src/armnnTfLiteParser/test/Split.cpp
             src/armnnTfLiteParser/test/SplitV.cpp
             src/armnnTfLiteParser/test/Squeeze.cpp
             src/armnnTfLiteParser/test/StridedSlice.cpp
             src/armnnTfLiteParser/test/Sub.cpp
             src/armnnTfLiteParser/test/Sum.cpp
             src/armnnTfLiteParser/test/TransposeConv.cpp
             src/armnnTfLiteParser/test/Transpose.cpp
             src/armnnTfLiteParser/test/Unpack.cpp
             src/armnnTfLiteParser/test/Unsupported.cpp
             src/armnnTfLiteParser/test/LoadModel.cpp
             src/armnnTfLiteParser/test/GetBuffer.cpp
             src/armnnTfLiteParser/test/OutputShapeOfSqueeze.cpp
             src/armnnTfLiteParser/test/InputOutputTensorNames.cpp
             src/armnnTfLiteParser/test/GetTensorIds.cpp
             src/armnnTfLiteParser/test/GetSubgraphInputsOutputs.cpp
             src/armnnTfLiteParser/test/GetInputsOutputs.cpp
             src/armnnTfLiteParser/test/TfLiteParser.cpp
             )

        # Generate SchemaText.cpp file which contains the TfLite schema text as a
        # static C-array of bytes. This is needed at runtime for TfLite parser tests.
        add_custom_command(
            OUTPUT  SchemaText.cpp
            COMMAND cp ${TF_LITE_SCHEMA_INCLUDE_PATH}/schema.fbs g_TfLiteSchemaText
            COMMAND xxd -i g_TfLiteSchemaText SchemaText.cpp
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${TF_LITE_SCHEMA_INCLUDE_PATH}/schema.fbs
        )
        list(APPEND unittest_sources ${CMAKE_CURRENT_BINARY_DIR}/SchemaText.cpp)
    endif()

    if(BUILD_ONNX_PARSER AND ARMNNREF)
        list(APPEND unittest_sources
            src/armnnOnnxParser/test/Addition.cpp
            src/armnnOnnxParser/test/BatchNorm.cpp
            src/armnnOnnxParser/test/Clip.cpp
            src/armnnOnnxParser/test/Concat.cpp
            src/armnnOnnxParser/test/Const.cpp
            src/armnnOnnxParser/test/Constructor.cpp
            src/armnnOnnxParser/test/Conv2D.cpp
            src/armnnOnnxParser/test/CreateNetwork.cpp
            src/armnnOnnxParser/test/DepthConv.cpp
            src/armnnOnnxParser/test/Flatten.cpp
            src/armnnOnnxParser/test/FullyConnected.cpp
            src/armnnOnnxParser/test/Gather.cpp
            src/armnnOnnxParser/test/Gemm.cpp
            src/armnnOnnxParser/test/GetInputsOutputs.cpp
            src/armnnOnnxParser/test/LoadScopeDynamicTensor.cpp
            src/armnnOnnxParser/test/OnnxParserTestUtils.cpp
            src/armnnOnnxParser/test/OnnxParserTestUtils.hpp
            src/armnnOnnxParser/test/Pooling.cpp
            src/armnnOnnxParser/test/ProtoxtFixture.cpp
            src/armnnOnnxParser/test/Relu.cpp
            src/armnnOnnxParser/test/Reshape.cpp
            src/armnnOnnxParser/test/Shape.cpp
            src/armnnOnnxParser/test/Unsqueeze.cpp
            )
    endif()

    if(BUILD_ARMNN_SERIALIZER AND ARMNNREF)
        enable_language(ASM)
        list(APPEND unittest_sources
            src/armnnSerializer/test/ActivationSerializationTests.cpp
            src/armnnSerializer/test/ComparisonSerializationTests.cpp
            src/armnnSerializer/test/LstmSerializationTests.cpp
            src/armnnSerializer/test/SerializerTests.cpp
            src/armnnSerializer/test/SerializerTestUtils.cpp
            src/armnnSerializer/test/SerializerTestUtils.hpp
            src/armnnDeserializer/test/DeserializeAbs.cpp
            src/armnnDeserializer/test/DeserializeActivation.cpp
            src/armnnDeserializer/test/DeserializeAdd.cpp
            src/armnnDeserializer/test/DeserializeArgMinMax.cpp
            src/armnnDeserializer/test/DeserializeBatchMatMul.cpp
            src/armnnDeserializer/test/DeserializeBatchToSpaceNd.cpp
            src/armnnDeserializer/test/DeserializeBatchNormalization.cpp
            src/armnnDeserializer/test/DeserializeCast.cpp
            src/armnnDeserializer/test/DeserializeChannelShuffle.cpp
            src/armnnDeserializer/test/DeserializeComparison.cpp
            src/armnnDeserializer/test/DeserializeConstant.cpp
            src/armnnDeserializer/test/DeserializeConvolution2d.cpp
            src/armnnDeserializer/test/DeserializeConvolution3d.cpp
            src/armnnDeserializer/test/DeserializeDepthToSpace.cpp
            src/armnnDeserializer/test/DeserializeDepthwiseConv2d.cpp
            src/armnnDeserializer/test/DeserializeDivision.cpp
            src/armnnDeserializer/test/DeserializeFill.cpp
            src/armnnDeserializer/test/DeserializeFloor.cpp
            src/armnnDeserializer/test/DeserializeFullyConnected.cpp
            src/armnnDeserializer/test/DeserializeGather.cpp
            src/armnnDeserializer/test/DeserializeGatherNd.cpp
            src/armnnDeserializer/test/DeserializeInstanceNormalization.cpp
            src/armnnDeserializer/test/DeserializeL2Normalization.cpp
            src/armnnDeserializer/test/DeserializeLogSoftmax.cpp
            src/armnnDeserializer/test/DeserializeMean.cpp
            src/armnnDeserializer/test/DeserializeMultiplication.cpp
            src/armnnDeserializer/test/DeserializeNormalization.cpp
            src/armnnDeserializer/test/DeserializePad.cpp
            src/armnnDeserializer/test/DeserializePermute.cpp
            src/armnnDeserializer/test/DeserializePooling2d.cpp
            src/armnnDeserializer/test/DeserializePooling3d.cpp
            src/armnnDeserializer/test/DeserializeRank.cpp
            src/armnnDeserializer/test/DeserializeReduceSum.cpp
            src/armnnDeserializer/test/DeserializeReshape.cpp
            src/armnnDeserializer/test/DeserializeResizeBilinear.cpp
            src/armnnDeserializer/test/DeserializeRsqrt.cpp
            src/armnnDeserializer/test/DeserializeShape.cpp
            src/armnnDeserializer/test/DeserializeSlice.cpp
            src/armnnDeserializer/test/DeserializeSpaceToBatchNd.cpp
            src/armnnDeserializer/test/DeserializeStridedSlice.cpp
            src/armnnDeserializer/test/DeserializeSubtraction.cpp
            src/armnnDeserializer/test/ParserFlatbuffersSerializeFixture.hpp
            src/armnnDeserializer/test/SchemaSerialize.s
            )
        set_source_files_properties(src/armnnDeserializer/test/SchemaSerialize.s PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
    endif()

    if(BUILD_GATORD_MOCK)
        list(APPEND unittest_sources
            tests/profiling/gatordmock/tests/GatordMockTests.cpp
            )
    endif()

    if(BUILD_TIMELINE_DECODER)
        list(APPEND unittest_sources
             src/timelineDecoder/tests/JSONTimelineDecoderTests.cpp
             profiling/server/src/timelineDecoder/tests/TimelineTests.cpp
             )
    endif()

    if(BUILD_BASE_PIPE_SERVER)
        list(APPEND unittest_sources
                profiling/server/src/basePipeServer/tests/BasePipeServerTests.cpp
                )
    endif()

    if(BUILD_TESTS)
        list(APPEND unittest_sources
            ./tests/NetworkExecutionUtils/NetworkExecutionUtils.cpp
            ./tests/NetworkExecutionUtils/test/NetworkExecutionUtilsTests.cpp)
    endif()

    foreach(lib ${armnnUnitTestLibraries})
        message(STATUS "Adding object library dependency to UnitTests: ${lib}")
        list(APPEND unittest_sources $<TARGET_OBJECTS:${lib}>)
    endforeach()

    add_executable(UnitTests ${unittest_sources})
    target_include_directories(UnitTests PRIVATE src/armnn)
    target_include_directories(UnitTests PRIVATE src/armnnUtils)
    target_include_directories(UnitTests PRIVATE src/armnnTestUtils)
    target_include_directories(UnitTests PRIVATE src/backends)
    target_include_directories(UnitTests PRIVATE src/profiling)

    if(VALGRIND_FOUND)
        if(HEAP_PROFILING OR LEAK_CHECKING)
            message(STATUS "Valgrind is disabled for heap profiling and leak checking builds.")
        else()
            # Valgrind works with gperftools version number <= 2.4
            target_compile_definitions(UnitTests PRIVATE "WITH_VALGRIND=1")
        endif()
    endif()

    target_link_libraries(UnitTests ${CMAKE_THREAD_LIBS_INIT})

    target_link_libraries(UnitTests armnn)
    target_link_libraries(UnitTests armnnTestUtils)

    if(BUILD_GATORD_MOCK)
        target_link_libraries(UnitTests gatordMockService)
    endif()

    if(BUILD_TIMELINE_DECODER)
        target_link_libraries(UnitTests timelineDecoder)
        target_link_libraries(UnitTests timelineDecoderJson)
    endif()

    if(BUILD_TF_LITE_PARSER)
        target_include_directories(UnitTests SYSTEM PRIVATE "${TF_LITE_SCHEMA_INCLUDE_PATH}")
        target_include_directories(UnitTests SYSTEM PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
        target_link_libraries(UnitTests armnnTfLiteParser)
    endif()

    if(BUILD_ARMNN_SERIALIZER AND ARMNNREF)
        target_include_directories(UnitTests SYSTEM PRIVATE generated)
        target_include_directories(UnitTests SYSTEM PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/src/armnnSerializer)
        target_include_directories(UnitTests SYSTEM PRIVATE "${FLATBUFFERS_INCLUDE_PATH}")
        target_link_libraries(UnitTests armnnSerializer)
    endif()

    if(BUILD_ONNX_PARSER)
        target_link_libraries(UnitTests armnnOnnxParser)
    endif()

    if(BUILD_BASE_PIPE_SERVER)
        target_link_libraries(UnitTests armnnBasePipeServer)
    endif()

    addDllCopyCommands(UnitTests)
endif()

if (BUILD_ARMNN_SERIALIZER AND (BUILD_TF_LITE_PARSER OR BUILD_ONNX_PARSER) AND ARMNNREF)
    set(ArmnnConverter_sources
        src/armnnConverter/ArmnnConverter.cpp)

    add_executable_ex(ArmnnConverter ${ArmnnConverter_sources})
    target_include_directories(ArmnnConverter PRIVATE src/armnn)
    target_include_directories(ArmnnConverter PRIVATE src/armnnUtils)

    if(BUILD_ONNX_PARSER)
        target_link_libraries(ArmnnConverter armnnOnnxParser)
    endif()

    if(BUILD_TF_LITE_PARSER)
        target_link_libraries(ArmnnConverter armnnTfLiteParser)
    endif()

    target_link_libraries(ArmnnConverter armnnSerializer)

    target_link_libraries(ArmnnConverter armnn)
    target_link_libraries(ArmnnConverter ${CMAKE_THREAD_LIBS_INIT})
    addDllCopyCommands(ArmnnConverter)

    install(TARGETS ArmnnConverter
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if(BUILD_BASE_PIPE_SERVER)
    add_subdirectory(profiling/server/src/basePipeServer)
endif()

if(BUILD_TIMELINE_DECODER)
    add_subdirectory(profiling/server/src/timelineDecoder)
    add_subdirectory(src/timelineDecoder)
endif()

if(BUILD_GATORD_MOCK AND NOT BUILD_BASE_PIPE_SERVER)
    message(ERROR, "In order to build GatordMock you must set BUILD_BASE_PIPE_SERVER = YES")
endif()

if(BUILD_GATORD_MOCK AND NOT BUILD_TIMELINE_DECODER)
    message(ERROR, "In order to build GatordMock you must set BUILD_TIMELINE_DECODER = YES")
endif()

if(BUILD_GATORD_MOCK)
    set(gatord_mock_sources)
    list(APPEND gatord_mock_sources
        tests/profiling/gatordmock/CommandFileParser.hpp
        tests/profiling/gatordmock/CommandFileParser.cpp
        tests/profiling/gatordmock/CommandLineProcessor.hpp
        tests/profiling/gatordmock/CommandLineProcessor.cpp
        tests/profiling/gatordmock/GatordMockService.hpp
        tests/profiling/gatordmock/GatordMockService.cpp
        tests/profiling/gatordmock/PeriodicCounterCaptureCommandHandler.cpp
        tests/profiling/gatordmock/PeriodicCounterCaptureCommandHandler.hpp
        tests/profiling/gatordmock/PeriodicCounterSelectionResponseHandler.cpp
        tests/profiling/gatordmock/PeriodicCounterSelectionResponseHandler.hpp
        tests/profiling/gatordmock/StreamMetadataCommandHandler.cpp
        tests/profiling/gatordmock/StreamMetadataCommandHandler.hpp
        tests/profiling/gatordmock/StubCommandHandler.hpp
        )

    add_library_ex(gatordMockService STATIC ${gatord_mock_sources})

    target_include_directories(gatordMockService PRIVATE profiling/common/include
                                                         profiling/client/include
                                                         profiling/server/src/basePipeServer
                                                         src/armnnUtils
                                                         src/profiling
                                                         src/timelineDecoder)

    if(BUILD_UNIT_TESTS)
        target_include_directories(UnitTests PRIVATE tests/profiling/gatordmock)
        target_include_directories(UnitTests PRIVATE src/backends/backendsCommon/test)
    endif()

    add_executable_ex(GatordMock tests/profiling/gatordmock/GatordMockMain.cpp)

    target_include_directories(GatordMock PRIVATE profiling/common/include
                                                  profiling/client/include
                                                  profiling/server/src/basePipeServer
                                                  src/armnnUtils
                                                  src/profiling
                                                  src/timelineDecoder)

    target_link_libraries(GatordMock PUBLIC
                          armnn
                          armnnBasePipeServer
                          timelineDecoder
                          gatordMockService)

    if(Threads_FOUND AND (NOT ("${CMAKE_SYSTEM_NAME}" STREQUAL Android)))
        target_link_libraries(GatordMock PUBLIC ${CMAKE_THREAD_LIBS_INIT})
    endif()

endif()

####################################################
# export targets
set(armnn_export_targets)
list(APPEND armnn_export_targets
    armnn
    armnnUtils
)

install(
    TARGETS ${armnn_export_targets}
    EXPORT  armnn-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

####################################################
## Set export alias
set_target_properties(armnn
    PROPERTIES
    EXPORT_NAME Armnn
)

## Export target scrips
install(
    EXPORT      armnn-targets
    FILE        ArmnnTargets.cmake
    NAMESPACE   Armnn::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

####################################################
## Create ArmnnConfig.cmake
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR})

message(STATUS "CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_DIR}" )
message(STATUS "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}" )
configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/cmake/ArmnnConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ArmnnConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

####################################################
## Install Armnn config file
install(
    FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ArmnnConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

####################################################
## Export from build tree
export(
    EXPORT      armnn-targets
    FILE        ${CMAKE_CURRENT_BINARY_DIR}/ArmnnTargets.cmake
    NAMESPACE   Armnn::
)

add_library(Armnn::Armnn ALIAS armnn)
add_library(Armnn::armnnUtils ALIAS armnnUtils)

####################################################
## Build Python bindings
if (BUILD_PYTHON_WHL OR BUILD_PYTHON_SRC)
    add_subdirectory(python/pyarmnn)
endif()
