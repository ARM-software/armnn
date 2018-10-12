//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backends/cl/ClWorkloadFactory.hpp>
#include <backends/reference/RefWorkloadFactory.hpp>

#include <backends/test/MemCopyTestImpl.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ClMemCopy)

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndGpu)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndCpu)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndGpuWithSubtensors)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenGpuAndCpuWithSubtensors)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_SUITE_END()
