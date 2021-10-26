//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/IRuntime.hpp>
#include <armnn/Utils.hpp>
#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

#if defined(ARMNNREF_ENABLED)
#include <reference/RefBackend.hpp>
#endif

#if defined(ARMCOMPUTENEON_ENABLED)
#include <neon/NeonBackend.hpp>
#endif

#include <doctest/doctest.h>


// Sample implementation of IMemoryOptimizerStrategy..
class SampleMemoryOptimizerStrategy : public armnn::IMemoryOptimizerStrategy
{
public:
    SampleMemoryOptimizerStrategy() = default;

    std::string GetName() const
    {
        return std::string("SampleMemoryOptimizerStrategy");
    }

    armnn::MemBlockStrategyType GetMemBlockStrategyType() const
    {
        return armnn::MemBlockStrategyType::SingleAxisPacking;
    }

    std::vector<armnn::MemBin> Optimize(std::vector<armnn::MemBlock>& memBlocks)
    {
        std::vector<armnn::MemBin> memBins;
        memBins.reserve(memBlocks.size());
        return memBins;
    }
};

TEST_SUITE("CustomMemoryOptimizerStrategyTests")
{

// Only run this test if CpuRef is enabled
#if defined(ARMNNREF_ENABLED)
TEST_CASE("RefCustomMemoryOptimizerStrategyTest")
{
    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    auto customMemoryOptimizerStrategy = std::make_shared<SampleMemoryOptimizerStrategy>();
    options.m_MemoryOptimizerStrategyMap = {{"CpuRef", std::move(customMemoryOptimizerStrategy)}};
    IRuntimePtr run = IRuntime::Create(options);

    CHECK(!BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().size() == 1);
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().at(RefBackend::GetIdStatic()));
    auto optimizerStrategy = BackendRegistryInstance().GetMemoryOptimizerStrategies().at(RefBackend::GetIdStatic());
    CHECK(optimizerStrategy->GetName() == std::string("SampleMemoryOptimizerStrategy"));

    // De-register the memory optimizer
    BackendRegistryInstance().DeregisterMemoryOptimizerStrategy(RefBackend::GetIdStatic());
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());

}

TEST_CASE("CpuRefSetMemoryOptimizerStrategyTest")
{
    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    options.m_BackendOptions.emplace_back(
         BackendOptions{"CpuRef",
         {
             {"MemoryOptimizerStrategy", "ConstantMemoryStrategy"}
         }
    });

    IRuntimePtr run = IRuntime::Create(options);

    // ConstantMemoryStrategy should be registered for CpuRef
    CHECK(!BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().size() == 1);
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().at(RefBackend::GetIdStatic()));
    auto optimizerStrategy = BackendRegistryInstance().GetMemoryOptimizerStrategies().at(RefBackend::GetIdStatic());
    CHECK(optimizerStrategy->GetName() == std::string("ConstantMemoryStrategy"));
    armnn::BackendRegistryInstance().DeregisterMemoryOptimizerStrategy(RefBackend::GetIdStatic());
}

#endif

// Only run this test if CpuAcc is enabled
#if defined(ARMCOMPUTENEON_ENABLED)

TEST_CASE("CpuAccSetMemoryOptimizerStrategyTest")
{
    using namespace armnn;

    // Create ArmNN runtime
    IRuntime::CreationOptions options; // default options
    options.m_BackendOptions.emplace_back(
         BackendOptions{"CpuAcc",
         {
             {"MemoryOptimizerStrategy", "NotExistMemoryOptimizerStrategy"}
         }
    });

    IRuntimePtr run = IRuntime::Create(options);

    // No MemoryOptimizerStrategy should be registered..
    CHECK(armnn::BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());
    armnn::BackendRegistryInstance().DeregisterMemoryOptimizerStrategy(NeonBackend::GetIdStatic());
}

#endif

} // test suite CustomMemoryOptimizerStrategyTests