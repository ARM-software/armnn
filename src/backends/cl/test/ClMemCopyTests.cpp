//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClWorkloadFactoryHelper.hpp"

#include <cl/ClWorkloadFactory.hpp>
#include <aclCommon/test/MemCopyTestImpl.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ClMemCopy)

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndGpu)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndCpu)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndGpuWithSubtensors)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndCpuWithSubtensors)
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_SUITE_END()
