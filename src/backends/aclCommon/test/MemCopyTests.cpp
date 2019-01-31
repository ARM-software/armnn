//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <aclCommon/ArmComputeTensorUtils.hpp>
#include <aclCommon/test/MemCopyTestImpl.hpp>

#if defined(ARMCOMPUTECL_ENABLED) && defined(ARMCOMPUTENEON_ENABLED)
#include <cl/ClWorkloadFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>
#include <cl/test/ClWorkloadFactoryHelper.hpp>

#include <neon/NeonWorkloadFactory.hpp>
#include <neon/test/NeonWorkloadFactoryHelper.hpp>
#endif

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(MemCopyCommon)

BOOST_AUTO_TEST_CASE(AclTypeConversions)
{
    arm_compute::Strides strides(1, 2, 3, 4);
    armnn::TensorShape convertedStrides = armnn::armcomputetensorutils::GetStrides(strides);

    BOOST_TEST(convertedStrides[0] == 4);
    BOOST_TEST(convertedStrides[1] == 3);
    BOOST_TEST(convertedStrides[2] == 2);
    BOOST_TEST(convertedStrides[3] == 1);

    arm_compute::TensorShape shape(5, 6, 7, 8);
    armnn::TensorShape convertedshape = armnn::armcomputetensorutils::GetShape(shape);

    BOOST_TEST(convertedshape[0] == 8);
    BOOST_TEST(convertedshape[1] == 7);
    BOOST_TEST(convertedshape[2] == 6);
    BOOST_TEST(convertedshape[3] == 5);
}

BOOST_AUTO_TEST_SUITE_END()

#if defined(ARMCOMPUTECL_ENABLED) && defined(ARMCOMPUTENEON_ENABLED)

BOOST_FIXTURE_TEST_SUITE(MemCopyClNeon, ClContextControlFixture)

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndGpu)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndNeon)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndGpuWithSubtensors)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndNeonWithSubtensors)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_SUITE_END()

#endif
