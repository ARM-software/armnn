//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <cl/test/ClContextControlFixture.hpp>
#include <backendsCommon/test/JsonPrinterTestImpl.hpp>

#include <doctest/doctest.h>

#include <vector>

TEST_CASE_FIXTURE(ClProfilingContextControlFixture, "SoftmaxProfilerJsonPrinterGpuAccTest")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    RunSoftmaxProfilerJsonPrinterTest(backends);

}