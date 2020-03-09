//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestDynamicBackend.hpp"

#include <armnn/backends/IBackendInternal.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

constexpr const char* TestDynamicBackendId()
{
#if defined(VALID_TEST_DYNAMIC_BACKEND_1)

    return "ValidTestDynamicBackend";

#elif defined(VALID_TEST_DYNAMIC_BACKEND_2) || \
      defined(VALID_TEST_DYNAMIC_BACKEND_4) || \
      defined(INVALID_TEST_DYNAMIC_BACKEND_9)

    // This backend id is shared among different test dynamic backends for testing purposes:
    // the test dynamic backend 4 is actually a duplicate of the test dynamic backend 2 (with the same version),
    // the test dynamic backend 9 is actually a duplicate of the test dynamic backend 2 (but with a version
    // incompatible with the current Backend API)
    return "TestValid2";

#elif defined(VALID_TEST_DYNAMIC_BACKEND_3)

    // The test dynamic backend 3 is a different backend than the test dynamic backend 2
    return "TestValid3";

#elif defined(VALID_TEST_DYNAMIC_BACKEND_5)

    // The test dynamic backend 5 is a different backend than the test dynamic backend 2
    return "TestValid5";

#elif defined(INVALID_TEST_DYNAMIC_BACKEND_10)

    // Empty backend id
    return "";

#elif defined(INVALID_TEST_DYNAMIC_BACKEND_11)

    // "Unknown" backend id, "Unknown" is a reserved id in ArmNN
    return "Unknown";

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
        IgnoreUnused(memoryManager);
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
#if defined(INVALID_TEST_DYNAMIC_BACKEND_5) || \
    defined(INVALID_TEST_DYNAMIC_BACKEND_8)

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

#if defined(INVALID_TEST_DYNAMIC_BACKEND_7) || \
    defined(INVALID_TEST_DYNAMIC_BACKEND_8)

    *outMajor = 0;
    *outMinor = 7;

#else

    armnn::BackendVersion apiVersion = armnn::IBackendInternal::GetApiVersion();

    *outMajor = apiVersion.m_Major;

#if defined(INVALID_TEST_DYNAMIC_BACKEND_9)

    *outMinor = apiVersion.m_Minor + 1;

#else

    *outMinor = apiVersion.m_Minor;

#endif

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
