//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestDynamicBackend.hpp"

#include <backendsCommon/IBackendInternal.hpp>

constexpr const char* TestDynamicBackendId()
{
#if defined(VALID_TEST_DYNAMIC_BACKEND)

    return "ValidTestDynamicBackend";

#else

    return "InvalidTestDynamicBackend";

#endif
}

class TestDynamicBackend : public armnn::IBackendInternal
{
public:
    TestDynamicBackend()
        : m_BackendId(TestDynamicBackendId())
    {}

    const armnn::BackendId& GetId() const override
    {
        return m_BackendId;
    }
    IWorkloadFactoryPtr CreateWorkloadFactory(const IMemoryManagerSharedPtr& memoryManager) const override
    {
        return IWorkloadFactoryPtr{};
    }
    ILayerSupportSharedPtr GetLayerSupport() const override
    {
        return ILayerSupportSharedPtr{};
    }

private:
    armnn::BackendId m_BackendId;
};

const char* GetBackendId()
{
#if defined(INVALID_TEST_DYNAMIC_BACKEND_5)

    // Return an invalid backend id
    return nullptr;

#else

    // Return a valid backend id
    return TestDynamicBackendId();

#endif
}

void GetVersion(uint32_t* outMajor, uint32_t* outMinor)
{
    if (!outMajor || !outMinor)
    {
        return;
    }

#if defined(INVALID_TEST_DYNAMIC_BACKEND_7)

    *outMajor = 0;
    *outMinor = 7;

#else

    *outMajor = 1;
    *outMinor = 0;

#endif
}

void* BackendFactory()
{
#if defined(INVALID_TEST_DYNAMIC_BACKEND_6)

    // Return an invalid backend instance
    return nullptr;

#else

    // Return a non-null backend instance
    return new TestDynamicBackend();

#endif
}
