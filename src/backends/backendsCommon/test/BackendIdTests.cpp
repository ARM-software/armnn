//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>
#include <armnn/Types.hpp>

#include <boost/test/unit_test.hpp>

using namespace armnn;

BOOST_AUTO_TEST_SUITE(BackendIdTests)

BOOST_AUTO_TEST_CASE(CreateBackendIdFromCompute)
{
    BackendId fromCompute{Compute::GpuAcc};
    BOOST_TEST(fromCompute.Get() == GetComputeDeviceAsCString(Compute::GpuAcc));
}

BOOST_AUTO_TEST_CASE(CreateBackendIdVectorFromCompute)
{
    std::vector<BackendId> fromComputes = {Compute::GpuAcc, Compute::CpuRef};
    BOOST_TEST(fromComputes[0].Get() == GetComputeDeviceAsCString(Compute::GpuAcc));
    BOOST_TEST(fromComputes[1].Get() == GetComputeDeviceAsCString(Compute::CpuRef));
}

BOOST_AUTO_TEST_SUITE_END()
