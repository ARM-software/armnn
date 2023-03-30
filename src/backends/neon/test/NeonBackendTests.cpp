//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <neon/NeonBackend.hpp>
#include <neon/NeonTensorHandleFactory.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("NeonBackendTests")
{
TEST_CASE("NeonRegisterTensorHandleFactoriesMatchingImportFactoryId")
{
    auto neonBackend = std::make_unique<NeonBackend>();
    TensorHandleFactoryRegistry registry;
    neonBackend->RegisterTensorHandleFactories(registry);

    // When calling RegisterTensorHandleFactories, CopyAndImportFactoryPair is registered
    // Get matching import factory id correctly
    CHECK((registry.GetMatchingImportFactoryId(NeonTensorHandleFactory::GetIdStatic()) ==
           NeonTensorHandleFactory::GetIdStatic()));
}

TEST_CASE("NeonCreateWorkloadFactoryMatchingImportFactoryId")
{
    auto neonBackend = std::make_unique<NeonBackend>();
    TensorHandleFactoryRegistry registry;
    neonBackend->CreateWorkloadFactory(registry);

    // When calling CreateWorkloadFactory, CopyAndImportFactoryPair is registered
    // Get matching import factory id correctly
    CHECK((registry.GetMatchingImportFactoryId(NeonTensorHandleFactory::GetIdStatic()) ==
           NeonTensorHandleFactory::GetIdStatic()));
}

TEST_CASE("NeonCreateWorkloadFactoryWithOptionsMatchingImportFactoryId")
{
    auto neonBackend = std::make_unique<NeonBackend>();
    TensorHandleFactoryRegistry registry;
    ModelOptions modelOptions;
    neonBackend->CreateWorkloadFactory(registry, modelOptions);

    // When calling CreateWorkloadFactory with ModelOptions, CopyAndImportFactoryPair is registered
    // Get matching import factory id correctly
    CHECK((registry.GetMatchingImportFactoryId(NeonTensorHandleFactory::GetIdStatic()) ==
           NeonTensorHandleFactory::GetIdStatic()));
}
}
