//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/RefMemoryManager.hpp>

#include <doctest/doctest.h>

TEST_SUITE("RefMemoryManagerTests")
{
using namespace armnn;
using Pool = RefMemoryManager::Pool;

TEST_CASE("ManageOneThing")
{
    RefMemoryManager memoryManager;

    Pool* pool = memoryManager.Manage(10);

    CHECK(pool);

    memoryManager.Acquire();

    CHECK(memoryManager.GetPointer(pool) != nullptr); // Yields a valid pointer

    memoryManager.Release();
}

TEST_CASE("ManageTwoThings")
{
    RefMemoryManager memoryManager;

    Pool* pool1 = memoryManager.Manage(10);
    Pool* pool2 = memoryManager.Manage(5);

    CHECK(pool1);
    CHECK(pool2);

    memoryManager.Acquire();

    void *p1 = memoryManager.GetPointer(pool1);
    void *p2 = memoryManager.GetPointer(pool2);

    CHECK(p1);
    CHECK(p2);
    CHECK(p1 != p2);

    memoryManager.Release();
}

}
