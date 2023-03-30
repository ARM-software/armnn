//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/TensorHandleFactoryRegistry.hpp>
#include <cl/ClBackend.hpp>
#include <cl/ClTensorHandleFactory.hpp>
#include <cl/ClImportTensorHandleFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("ClBackendTests")
{
TEST_CASE("ClRegisterTensorHandleFactoriesMatchingImportFactoryId")
{
    auto clBackend = std::make_unique<ClBackend>();
    TensorHandleFactoryRegistry registry;
    clBackend->RegisterTensorHandleFactories(registry);

    // When calling RegisterTensorHandleFactories, CopyAndImportFactoryPair is registered
    // Get ClImportTensorHandleFactory id as the matching import factory id
    CHECK((registry.GetMatchingImportFactoryId(ClTensorHandleFactory::GetIdStatic()) ==
           ClImportTensorHandleFactory::GetIdStatic()));
}

TEST_CASE("ClRegisterTensorHandleFactoriesWithMemorySourceFlagsMatchingImportFactoryId")
{
    auto clBackend = std::make_unique<ClBackend>();
    TensorHandleFactoryRegistry registry;
    clBackend->RegisterTensorHandleFactories(registry,
                                             static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                             static_cast<MemorySourceFlags>(MemorySource::Malloc));

    // When calling RegisterTensorHandleFactories with MemorySourceFlags, CopyAndImportFactoryPair is registered
    // Get ClImportTensorHandleFactory id as the matching import factory id
    CHECK((registry.GetMatchingImportFactoryId(ClTensorHandleFactory::GetIdStatic()) ==
           ClImportTensorHandleFactory::GetIdStatic()));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClCreateWorkloadFactoryMatchingImportFactoryId")
{
    auto clBackend = std::make_unique<ClBackend>();
    TensorHandleFactoryRegistry registry;
    clBackend->CreateWorkloadFactory(registry);

    // When calling CreateWorkloadFactory, CopyAndImportFactoryPair is registered
    // Get ClImportTensorHandleFactory id as the matching import factory id
    CHECK((registry.GetMatchingImportFactoryId(ClTensorHandleFactory::GetIdStatic()) ==
           ClImportTensorHandleFactory::GetIdStatic()));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClCreateWorkloadFactoryWithOptionsMatchingImportFactoryId")
{
    auto clBackend = std::make_unique<ClBackend>();
    TensorHandleFactoryRegistry registry;
    ModelOptions modelOptions;
    clBackend->CreateWorkloadFactory(registry, modelOptions);

    // When calling CreateWorkloadFactory with ModelOptions, CopyAndImportFactoryPair is registered
    // Get ClImportTensorHandleFactory id as the matching import factory id
    CHECK((registry.GetMatchingImportFactoryId(ClTensorHandleFactory::GetIdStatic()) ==
           ClImportTensorHandleFactory::GetIdStatic()));
}

TEST_CASE_FIXTURE(ClContextControlFixture, "ClCreateWorkloadFactoryWitMemoryFlagsMatchingImportFactoryId")
{
    auto clBackend = std::make_unique<ClBackend>();
    TensorHandleFactoryRegistry registry;
    ModelOptions modelOptions;
    clBackend->CreateWorkloadFactory(registry, modelOptions,
                                     static_cast<MemorySourceFlags>(MemorySource::Malloc),
                                     static_cast<MemorySourceFlags>(MemorySource::Malloc));

    // When calling CreateWorkloadFactory with ModelOptions and MemorySourceFlags,
    // CopyAndImportFactoryPair is registered
    // Get ClImportTensorHandleFactory id as the matching import factory id
    CHECK((registry.GetMatchingImportFactoryId(ClTensorHandleFactory::GetIdStatic()) ==
           ClImportTensorHandleFactory::GetIdStatic()));
}
}
