//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backends/test/EndToEndTestImpl.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ClEndToEnd)

BOOST_AUTO_TEST_CASE(ConstantUsage_Cl_Float32)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    ConstantUsageFloat32Test(backends);
}

BOOST_AUTO_TEST_SUITE_END()
