//
// Copyright © 2021, 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <UnitTests.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include <neon/NeonWorkloadFactory.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Compute_ArmComputeNeon")
{
using namespace armnn;

using FactoryType = NeonWorkloadFactory;

// ============================================================================
// This is a specific subset of NeonLayerTests that can fail because of a known problem
// in the Android NDK. https://github.com/android/ndk/issues/1135
// We extract them here so then in the case of a debug Android build they can be excluded.
// The tests will pass in a release build. The problem has been corrected in NDK r21.

// Softmax
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)

// LogSoftmax
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_1, LogSoftmaxTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_2, LogSoftmaxTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_3, LogSoftmaxTest3<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_4, LogSoftmaxTest4<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxInt8_1, LogSoftmaxTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxInt8_2, LogSoftmaxTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxInt8_3, LogSoftmaxTest3<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxInt8_4, LogSoftmaxTest4<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxUint8_1, LogSoftmaxTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxUint8_2, LogSoftmaxTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxUint8_3, LogSoftmaxTest3<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxUint8_4, LogSoftmaxTest4<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dNhwc, L2Normalization1dTest, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNorm,
                              LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNorm,
                              UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNormTest)

// ReduceSum
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumFloat32, ReduceSumSimpleTest<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumSingleAxisFloat32_3, ReduceSumSingleAxisTest3<DataType::Float32>)

#if defined(ARMNNREF_ENABLED)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta1WithReference, CompareSoftmaxTest, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta2WithReference, CompareSoftmaxTest, 2.0f)

#endif

}
