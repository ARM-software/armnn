//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <backendsCommon/test/JsonPrinterTestImpl.hpp>

#include <doctest/doctest.h>

#include <vector>

TEST_SUITE("NeonJsonPrinter")
{
TEST_CASE("SoftmaxProfilerJsonPrinterCpuAccTest")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    RunSoftmaxProfilerJsonPrinterTest(backends);
}

}