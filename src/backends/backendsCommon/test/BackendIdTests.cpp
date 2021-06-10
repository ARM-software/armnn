//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>
#include <armnn/Types.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("BackendIdTests")
{
TEST_CASE("CreateBackendIdFromCompute")
{
    BackendId fromCompute{Compute::GpuAcc};
    CHECK(fromCompute.Get() == GetComputeDeviceAsCString(Compute::GpuAcc));
}

TEST_CASE("CreateBackendIdVectorFromCompute")
{
    std::vector<BackendId> fromComputes = {Compute::GpuAcc, Compute::CpuRef};
    CHECK(fromComputes[0].Get() == GetComputeDeviceAsCString(Compute::GpuAcc));
    CHECK(fromComputes[1].Get() == GetComputeDeviceAsCString(Compute::CpuRef));
}

}
