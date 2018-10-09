//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>

#include <backends/BackendRegistry.hpp>
#include <armnn/Types.hpp>

namespace
{

class SwapRegistryStorage : public armnn::BackendRegistry
{
public:
    SwapRegistryStorage() : armnn::BackendRegistry()
    {
        Swap(m_TempStorage);
    }

    ~SwapRegistryStorage()
    {
        Swap(m_TempStorage);
    }

private:
    BackendRegistry::FactoryStorage m_TempStorage;
};

}

BOOST_AUTO_TEST_SUITE(BackendRegistryTests)

BOOST_AUTO_TEST_CASE(SwapRegistry)
{
    using armnn::BackendRegistry;
    auto nFactories = BackendRegistry::Instance().Size();
    {
        SwapRegistryStorage helper;
        BOOST_TEST(BackendRegistry::Instance().Size() == 0);
    }
    BOOST_TEST(BackendRegistry::Instance().Size() == nFactories);
}

BOOST_AUTO_TEST_CASE(TestRegistryHelper)
{
    using armnn::BackendRegistry;
    SwapRegistryStorage helper;

    bool called = false;
    BackendRegistry::Helper factoryHelper("HelloWorld", [&called]() {
        called = true;
        return armnn::IBackendUniquePtr(nullptr, nullptr);
    } );

    // sanity check: the factory has not been called yet
    BOOST_TEST(called == false);

    auto factoryFunction = BackendRegistry::Instance().GetFactory("HelloWorld");

    // sanity check: the factory still not called
    BOOST_TEST(called == false);

    factoryFunction();
    BOOST_TEST(called == true);
}

BOOST_AUTO_TEST_CASE(TestDirectCallToRegistry)
{
    using armnn::BackendRegistry;
    SwapRegistryStorage helper;

    bool called = false;
    BackendRegistry::Instance().Register("HelloWorld", [&called]() {
        called = true;
        return armnn::IBackendUniquePtr(nullptr, nullptr);
    } );

    // sanity check: the factory has not been called yet
    BOOST_TEST(called == false);

    auto factoryFunction = BackendRegistry::Instance().GetFactory("HelloWorld");

    // sanity check: the factory still not called
    BOOST_TEST(called == false);

    factoryFunction();
    BOOST_TEST(called == true);
}

BOOST_AUTO_TEST_SUITE_END()
