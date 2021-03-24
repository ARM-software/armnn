//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <test/TensorHelpers.hpp>
#include <test/UnitTests.hpp>

#include <neon/NeonLayerSupport.hpp>
#include <neon/NeonWorkloadFactory.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <backendsCommon/test/ActivationFixture.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Compute_ArmComputeNeon)

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

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dNhwc, L2Normalization1dTest, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNorm,
                              LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta1WithReference, CompareSoftmaxTest, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta2WithReference, CompareSoftmaxTest, 2.0f)

// ReduceSum
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumFloat32, ReduceSumSimpleTest<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumSingleAxisFloat32_3, ReduceSumSingleAxisTest3<DataType::Float32>)


BOOST_AUTO_TEST_SUITE_END()
