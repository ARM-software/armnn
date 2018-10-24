//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <backends/cl/test/ClContextControlFixture.hpp>
#include <backends/test/JsonPrinterTestImpl.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_FIXTURE_TEST_SUITE(ClJsonPrinter, ClProfilingContextControlFixture)

BOOST_AUTO_TEST_CASE(SoftmaxProfilerJsonPrinterGpuAccTest)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    SetupSoftmaxProfilerWithSpecifiedBackendsAndValidateJsonPrinterResult(backends);
}

BOOST_AUTO_TEST_SUITE_END()