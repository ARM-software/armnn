//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>
#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IBackendInternal.hpp>
#include <backendsCommon/memoryOptimizerStrategyLibrary/strategies/ConstantMemoryStrategy.hpp>
#include <reference/RefBackend.hpp>

#include <doctest/doctest.h>

namespace
{

class SwapRegistryStorage : public armnn::BackendRegistry
{
public:
    SwapRegistryStorage() : armnn::BackendRegistry()
    {
        Swap(armnn::BackendRegistryInstance(),  m_TempStorage);
    }

    ~SwapRegistryStorage()
    {
        Swap(armnn::BackendRegistryInstance(),m_TempStorage);
    }

private:
    FactoryStorage m_TempStorage;
};

}

TEST_SUITE("BackendRegistryTests")
{
TEST_CASE("SwapRegistry")
{
    using namespace armnn;
    auto nFactories = BackendRegistryInstance().Size();
    {
        SwapRegistryStorage helper;
        CHECK(BackendRegistryInstance().Size() == 0);
    }
    CHECK(BackendRegistryInstance().Size() == nFactories);
}

TEST_CASE("TestRegistryHelper")
{
    using namespace armnn;
    SwapRegistryStorage helper;

    bool called = false;

    BackendRegistry::StaticRegistryInitializer factoryHelper(
        BackendRegistryInstance(),
        "HelloWorld",
        [&called]()
        {
            called = true;
            return armnn::IBackendInternalUniquePtr(nullptr);
        }
    );

    // sanity check: the factory has not been called yet
    CHECK(called == false);

    auto factoryFunction = BackendRegistryInstance().GetFactory("HelloWorld");

    // sanity check: the factory still not called
    CHECK(called == false);

    factoryFunction();
    CHECK(called == true);
    BackendRegistryInstance().Deregister("HelloWorld");
}

TEST_CASE("TestDirectCallToRegistry")
{
    using namespace armnn;
    SwapRegistryStorage helper;

    bool called = false;
    BackendRegistryInstance().Register(
        "HelloWorld",
        [&called]()
        {
            called = true;
            return armnn::IBackendInternalUniquePtr(nullptr);
        }
    );

    // sanity check: the factory has not been called yet
    CHECK(called == false);

    auto factoryFunction = BackendRegistryInstance().GetFactory("HelloWorld");

    // sanity check: the factory still not called
    CHECK(called == false);

    factoryFunction();
    CHECK(called == true);
    BackendRegistryInstance().Deregister("HelloWorld");
}

// Test that backends can throw exceptions during their factory function to prevent loading in an unsuitable
// environment. For example Neon Backend loading on armhf device without neon support.
// In reality the dynamic backend is loaded in during the LoadDynamicBackends(options.m_DynamicBackendsPath)
// step of runtime constructor, then the factory function is called to check if supported, in case
// of Neon not being detected the exception is raised and so the backend is not added to the supportedBackends
// list

TEST_CASE("ThrowBackendUnavailableException")
{
    using namespace armnn;

    const BackendId mockBackendId("MockDynamicBackend");

    const std::string exceptionMessage("Mock error message to test unavailable backend");

    // Register the mock backend with a factory function lambda that always throws
    BackendRegistryInstance().Register(mockBackendId,
            [exceptionMessage]()
            {
                throw armnn::BackendUnavailableException(exceptionMessage);
                return IBackendInternalUniquePtr(); // Satisfy return type
            });

    // Get the factory function of the mock backend
    auto factoryFunc = BackendRegistryInstance().GetFactory(mockBackendId);

    try
    {
        // Call the factory function as done during runtime backend registering
        auto backend = factoryFunc();
        FAIL("Expected exception to have been thrown");
    }
    catch (const BackendUnavailableException& e)
    {
        // Caught
        CHECK_EQ(e.what(), exceptionMessage);
    }
    // Clean up the registry for the next test.
    BackendRegistryInstance().Deregister(mockBackendId);
}

#if defined(ARMNNREF_ENABLED)
TEST_CASE("RegisterMemoryOptimizerStrategy")
{
    using namespace armnn;

    const BackendId cpuRefBackendId(armnn::Compute::CpuRef);
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());

    // Register the memory optimizer
    std::shared_ptr<IMemoryOptimizerStrategy> memoryOptimizerStrategy =
        std::make_shared<ConstantMemoryStrategy>();
    BackendRegistryInstance().RegisterMemoryOptimizerStrategy(cpuRefBackendId, memoryOptimizerStrategy);
    CHECK(!BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().size() == 1);
    // De-register the memory optimizer
    BackendRegistryInstance().DeregisterMemoryOptimizerStrategy(cpuRefBackendId);
    CHECK(BackendRegistryInstance().GetMemoryOptimizerStrategies().empty());
}
#endif

}
