//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backends/neon/NeonWorkloadFactory.hpp>
#include <backends/reference/RefWorkloadFactory.hpp>
#include <backends/aclCommon/test/MemCopyTestImpl.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(NeonMemCopy)

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndNeon)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndCpu)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory>(false);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenCpuAndNeonWithSubtensors)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_CASE(CopyBetweenNeonAndCpuWithSubtensors)
{
    LayerTestResult<float, 4> result = MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory>(true);
    BOOST_TEST(CompareTensors(result.output, result.outputExpected));
}

BOOST_AUTO_TEST_SUITE_END()
