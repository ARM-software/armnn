//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/BackendId.hpp>

#include <cl/test/ClContextControlFixture.hpp>
#include <backendsCommon/test/JsonPrinterTestImpl.hpp>

#include <boost/test/unit_test.hpp>

#include <vector>

BOOST_FIXTURE_TEST_SUITE(ClJsonPrinter, ClProfilingContextControlFixture)

BOOST_AUTO_TEST_CASE(SoftmaxProfilerJsonPrinterGpuAccTest)
{
    std::vector<armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    RunSoftmaxProfilerJsonPrinterTest(backends);
}

BOOST_AUTO_TEST_SUITE_END()