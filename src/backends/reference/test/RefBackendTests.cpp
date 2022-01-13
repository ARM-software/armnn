//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/RefBackend.hpp>
#include <reference/RefTensorHandleFactory.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("RefBackendTests")
{
TEST_CASE("RefRegisterTensorHandleFactoriesMatchingImportFactoryId")
{
    auto refBackend = std::make_unique<RefBackend>();
    TensorHandleFactoryRegistry registry;
    refBackend->RegisterTensorHandleFactories(registry);

    // When calling RegisterTensorHandleFactories, CopyAndImportFactoryPair is registered
    // Get matching import factory id correctly
    CHECK((registry.GetMatchingImportFactoryId(RefTensorHandleFactory::GetIdStatic()) ==
           RefTensorHandleFactory::GetIdStatic()));
}

TEST_CASE("RefCreateWorkloadFactoryMatchingImportFactoryId")
{
    auto refBackend = std::make_unique<RefBackend>();
    TensorHandleFactoryRegistry registry;
    refBackend->CreateWorkloadFactory(registry);

    // When calling CreateWorkloadFactory, CopyAndImportFactoryPair is registered
    // Get matching import factory id correctly
    CHECK((registry.GetMatchingImportFactoryId(RefTensorHandleFactory::GetIdStatic()) ==
           RefTensorHandleFactory::GetIdStatic()));
}
}
