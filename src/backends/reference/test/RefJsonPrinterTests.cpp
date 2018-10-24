//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <backends/test/JsonPrinterTestImpl.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_AUTO_TEST_SUITE(RefJsonPrinter)

BOOST_AUTO_TEST_CASE(SoftmaxProfilerJsonPrinterCpuRefTest)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::CpuRef};
    SetupSoftmaxProfilerWithSpecifiedBackendsAndValidateJsonPrinterResult(backends);
}

BOOST_AUTO_TEST_SUITE_END()