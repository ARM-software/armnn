//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <reference/RefTensorHandle.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefTensorHandleTests)
using namespace armnn;

BOOST_AUTO_TEST_CASE(AcquireAndRelease)
{
    std::shared_ptr<RefMemoryManager> memoryManager = std::make_shared<RefMemoryManager>();

    TensorInfo info({1,1,1,1}, DataType::Float32);
    RefTensorHandle handle(info, memoryManager);

    handle.Manage();
    handle.Allocate();

    memoryManager->Acquire();
    {
        float *buffer = reinterpret_cast<float *>(handle.Map());

        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 2.5f;

        BOOST_CHECK(buffer[0] == 2.5f); // Memory is writable and readable

    }
    memoryManager->Release();

    memoryManager->Acquire();
    {
        float *buffer = reinterpret_cast<float *>(handle.Map());

        BOOST_CHECK(buffer != nullptr); // Yields a valid pointer

        buffer[0] = 3.5f;

        BOOST_CHECK(buffer[0] == 3.5f); // Memory is writable and readable
    }
    memoryManager->Release();
}

BOOST_AUTO_TEST_SUITE_END()