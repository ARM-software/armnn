//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Types.hpp>
#include <armnn/BackendRegistry.hpp>

#include <armnn/backends/IBackendInternal.hpp>

#include <boost/test/unit_test.hpp>

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

BOOST_AUTO_TEST_SUITE(BackendRegistryTests)

BOOST_AUTO_TEST_CASE(SwapRegistry)
{
    using namespace armnn;
    auto nFactories = BackendRegistryInstance().Size();
    {
        SwapRegistryStorage helper;
        BOOST_TEST(BackendRegistryInstance().Size() == 0);
    }
    BOOST_TEST(BackendRegistryInstance().Size() == nFactories);
}

BOOST_AUTO_TEST_CASE(TestRegistryHelper)
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
    BOOST_TEST(called == false);

    auto factoryFunction = BackendRegistryInstance().GetFactory("HelloWorld");

    // sanity check: the factory still not called
    BOOST_TEST(called == false);

    factoryFunction();
    BOOST_TEST(called == true);
    BackendRegistryInstance().Deregister("HelloWorld");
}

BOOST_AUTO_TEST_CASE(TestDirectCallToRegistry)
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
    BOOST_TEST(called == false);

    auto factoryFunction = BackendRegistryInstance().GetFactory("HelloWorld");

    // sanity check: the factory still not called
    BOOST_TEST(called == false);

    factoryFunction();
    BOOST_TEST(called == true);
    BackendRegistryInstance().Deregister("HelloWorld");
}

BOOST_AUTO_TEST_SUITE_END()
