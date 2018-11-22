//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <backendsCommon/test/JsonPrinterTestImpl.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(NeonJsonPrinter)

BOOST_AUTO_TEST_CASE(SoftmaxProfilerJsonPrinterCpuAccTest)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    RunSoftmaxProfilerJsonPrinterTest(backends);
}

BOOST_AUTO_TEST_SUITE_END()