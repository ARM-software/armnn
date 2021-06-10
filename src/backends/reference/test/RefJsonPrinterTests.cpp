//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <backendsCommon/test/JsonPrinterTestImpl.hpp>

#include <doctest/doctest.h>

#include <vector>

TEST_SUITE("RefJsonPrinter")
{
TEST_CASE("SoftmaxProfilerJsonPrinterCpuRefTest")
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    RunSoftmaxProfilerJsonPrinterTest(backends);
}

}