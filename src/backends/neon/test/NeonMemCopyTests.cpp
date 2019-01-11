//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <aclCommon/test/MemCopyTestImpl.hpp>

#include <neon/NeonWorkloadFactory.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeonMemCopy)

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndNeon)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndCpu)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndNeonWithSubtensors)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndCpuWithSubtensors)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_SUITE_END()
