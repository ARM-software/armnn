#
# Copyright © 2017 ARM Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

# COMMON_SOURCES contains the list of files to be included
# in the Android build and it is picked up by the Android.mk
# file in the root of ArmNN

COMMON_SOURCES := \
    TensorHandle.cpp \
    DynamicBackend.cpp \
    DynamicBackendUtils.cpp \
    IBackendInternal.cpp \
    ITensorHandleFactory.cpp \
    LayerSupportBase.cpp \
    MapWorkload.cpp \
    MemCopyWorkload.cpp \
    MemImportWorkload.cpp \
    MemoryManager.cpp \
    MemSyncWorkload.cpp \
    OptimizationViews.cpp \
    TensorHandleFactoryRegistry.cpp \
    UnmapWorkload.cpp \
    WorkloadData.cpp \
    WorkloadFactory.cpp \
    WorkloadUtils.cpp \
    memoryOptimizerStrategyLibrary/strategies/ConstantMemoryStrategy.cpp \
	memoryOptimizerStrategyLibrary/strategies/SingleAxisPriorityList.cpp \
    memoryOptimizerStrategyLibrary/strategies/StrategyValidator.cpp


# COMMON_TEST_SOURCES contains the list of files to be included
# in the Android unit test build (armnn-tests) and it is picked
# up by the Android.mk file in the root of ArmNN

COMMON_TEST_SOURCES := \
    test/CustomMemoryOptimizerStrategyTests.cpp \
    test/InstanceNormalizationEndToEndTestImpl.cpp \
    test/JsonPrinterTestImpl.cpp \
    test/LogSoftmaxEndToEndTestImpl.cpp \
    test/QLstmEndToEndTestImpl.cpp \
    test/QuantizedLstmEndToEndTestImpl.cpp \
    test/SpaceToDepthEndToEndTestImpl.cpp \
    test/layerTests/AbsTestImpl.cpp \
    test/layerTests/ActivationTestImpl.cpp \
    test/layerTests/AdditionTestImpl.cpp \
    test/layerTests/ArgMinMaxTestImpl.cpp \
    test/layerTests/BatchNormalizationTestImpl.cpp \
    test/layerTests/CastTestImpl.cpp \
    test/layerTests/ChannelShuffleTestImpl.cpp \
    test/layerTests/ComparisonTestImpl.cpp \
    test/layerTests/ConcatTestImpl.cpp \
    test/layerTests/ConstantTestImpl.cpp \
    test/layerTests/Conv2dTestImpl.cpp \
    test/layerTests/Conv3dTestImpl.cpp \
    test/layerTests/ConvertBf16ToFp32TestImpl.cpp \
    test/layerTests/ConvertFp16ToFp32TestImpl.cpp \
    test/layerTests/ConvertFp32ToBf16TestImpl.cpp \
    test/layerTests/ConvertFp32ToFp16TestImpl.cpp \
    test/layerTests/DebugTestImpl.cpp \
    test/layerTests/DepthToSpaceTestImpl.cpp \
    test/layerTests/DequantizeTestImpl.cpp \
    test/layerTests/DivisionTestImpl.cpp \
    test/layerTests/ElementwiseUnaryTestImpl.cpp \
    test/layerTests/ExpTestImpl.cpp \
    test/layerTests/FakeQuantizationTestImpl.cpp \
    test/layerTests/FillTestImpl.cpp \
    test/layerTests/FloorTestImpl.cpp \
    test/layerTests/FullyConnectedTestImpl.cpp \
    test/layerTests/GatherTestImpl.cpp \
    test/layerTests/InstanceNormalizationTestImpl.cpp \
    test/layerTests/L2NormalizationTestImpl.cpp \
    test/layerTests/LogTestImpl.cpp \
    test/layerTests/LogicalTestImpl.cpp \
    test/layerTests/LogSoftmaxTestImpl.cpp \
    test/layerTests/LstmTestImpl.cpp \
    test/layerTests/MaximumTestImpl.cpp \
    test/layerTests/MinimumTestImpl.cpp \
    test/layerTests/MirrorPadTestImpl.cpp \
    test/layerTests/MultiplicationTestImpl.cpp \
    test/layerTests/NegTestImpl.cpp \
    test/layerTests/NormalizationTestImpl.cpp \
    test/layerTests/PadTestImpl.cpp \
    test/layerTests/Pooling2dTestImpl.cpp \
    test/layerTests/Pooling3dTestImpl.cpp \
    test/layerTests/RankTestImpl.cpp \
    test/layerTests/ReductionTestImpl.cpp \
    test/layerTests/ReduceProdTestImpl.cpp \
    test/layerTests/ReduceSumTestImpl.cpp \
    test/layerTests/ReshapeTestImpl.cpp \
    test/layerTests/ResizeTestImpl.cpp \
    test/layerTests/RsqrtTestImpl.cpp \
    test/layerTests/SliceTestImpl.cpp \
    test/layerTests/QuantizeTestImpl.cpp \
    test/layerTests/SinTestImpl.cpp \
    test/layerTests/ShapeTestImpl.cpp \
    test/layerTests/SoftmaxTestImpl.cpp \
    test/layerTests/SpaceToBatchNdTestImpl.cpp \
    test/layerTests/SpaceToDepthTestImpl.cpp \
    test/layerTests/SplitterTestImpl.cpp \
    test/layerTests/StackTestImpl.cpp \
    test/layerTests/StridedSliceTestImpl.cpp \
    test/layerTests/SubtractionTestImpl.cpp \
    test/layerTests/TransposeConvolution2dTestImpl.cpp \
    test/layerTests/UnidirectionalSequenceLstmTestImpl.cpp \
    memoryOptimizerStrategyLibrary/test/ConstMemoryStrategyTests.cpp \
    memoryOptimizerStrategyLibrary/test/ValidatorStrategyTests.cpp \
    memoryOptimizerStrategyLibrary/test/SingleAxisPriorityListTests.cpp

ifeq ($(ARMNN_REF_ENABLED),1)
COMMON_TEST_SOURCES += \
    test/WorkloadDataValidation.cpp
endif # ARMNN_REF_ENABLED == 1
