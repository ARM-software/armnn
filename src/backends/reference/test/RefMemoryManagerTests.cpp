//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/RefMemoryManager.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefMemoryManagerTests)
using namespace armnn;
using Pool = RefMemoryManager::Pool;

BOOST_AUTO_TEST_CASE(ManageOneThing)
{
    RefMemoryManager memoryManager;

    Pool* pool = memoryManager.Manage(10);

    BOOST_CHECK(pool);

    memoryManager.Acquire();

    BOOST_CHECK(memoryManager.GetPointer(pool) != nullptr); // Yields a valid pointer

    memoryManager.Release();
}

BOOST_AUTO_TEST_CASE(ManageTwoThings)
{
    RefMemoryManager memoryManager;

    Pool* pool1 = memoryManager.Manage(10);
    Pool* pool2 = memoryManager.Manage(5);

    BOOST_CHECK(pool1);
    BOOST_CHECK(pool2);

    memoryManager.Acquire();

    void *p1 = memoryManager.GetPointer(pool1);
    void *p2 = memoryManager.GetPointer(pool2);

    BOOST_CHECK(p1);
    BOOST_CHECK(p2);
    BOOST_CHECK(p1 != p2);

    memoryManager.Release();
}

BOOST_AUTO_TEST_SUITE_END()
