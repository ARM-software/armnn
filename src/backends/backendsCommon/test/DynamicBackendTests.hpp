//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/DynamicBackend.hpp>
#include <backendsCommon/DynamicBackendUtils.hpp>

#include <string>
#include <memory>

#include <string>

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>

static std::string g_TestSharedObjectSubDir                 = "src/backends/backendsCommon/test/";
static std::string g_TestSharedObjectFileName               = "libarmnnTestSharedObject.so";
static std::string g_TestValidTestDynamicBackendFileName    = "libarmnnValidTestDynamicBackend.so";
static std::string g_TestInvalidTestDynamicBackend1FileName = "libarmnnInvalidTestDynamicBackend1.so";
static std::string g_TestInvalidTestDynamicBackend2FileName = "libarmnnInvalidTestDynamicBackend2.so";
static std::string g_TestInvalidTestDynamicBackend3FileName = "libarmnnInvalidTestDynamicBackend3.so";
static std::string g_TestInvalidTestDynamicBackend4FileName = "libarmnnInvalidTestDynamicBackend4.so";
static std::string g_TestInvalidTestDynamicBackend5FileName = "libarmnnInvalidTestDynamicBackend5.so";
static std::string g_TestInvalidTestDynamicBackend6FileName = "libarmnnInvalidTestDynamicBackend6.so";
static std::string g_TestInvalidTestDynamicBackend7FileName = "libarmnnInvalidTestDynamicBackend7.so";

std::string GetTestFilePath(const std::string& fileName)
{
    using namespace boost::filesystem;

    path currentPath(current_path());
    path sharedObjectPath = currentPath.append(g_TestSharedObjectSubDir);
    path sharedObjectFile = sharedObjectPath.append(fileName);
    BOOST_TEST(exists(sharedObjectFile));

    return sharedObjectFile.string();
}

void OpenCloseHandleTestImpl()
{
    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void CloseInvalidHandleTestImpl()
{
    using namespace armnn;

    // This calls must silently handle invalid handles and complete successfully (no segfaults, etc.)
    DynamicBackendUtils::CloseHandle(nullptr);
}

void OpenEmptyFileNameTestImpl()
{
    using namespace armnn;

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(""), RuntimeException);
    BOOST_TEST((sharedObjectHandle == nullptr));
}

void OpenNotExistingFileTestImpl()
{
    using namespace armnn;

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle("NotExistingFileName"), RuntimeException);
    BOOST_TEST((sharedObjectHandle == nullptr));
}

void OpenNotSharedObjectTestImpl()
{
    using namespace armnn;

    std::string notSharedObjectFilePath = GetTestFilePath("libarmnnNoSharedObject.txt");

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(notSharedObjectFilePath), RuntimeException);
    BOOST_TEST((sharedObjectHandle == nullptr));
}

void GetValidEntryPointTestImpl()
{
    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    BOOST_CHECK_NO_THROW(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                    "TestFunction1"));
    BOOST_TEST((testFunctionPointer != nullptr));
    BOOST_TEST(testFunctionPointer(7) == 7);

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void GetNameMangledEntryPointTestImpl()
{
    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    BOOST_CHECK_THROW(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                 "TestFunction2"),
                      RuntimeException);
    BOOST_TEST((testFunctionPointer == nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void GetNoExternEntryPointTestImpl()
{
    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    BOOST_CHECK_THROW(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                 "TestFunction3"),
                      RuntimeException);
    BOOST_TEST((testFunctionPointer == nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void GetNotExistingEntryPointTestImpl()
{
    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    BOOST_CHECK_THROW(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                 "TestFunction4"),
                      RuntimeException);
    BOOST_TEST((testFunctionPointer == nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void BackendVersioningTestImpl()
{
    using namespace armnn;

    class TestDynamicBackendUtils : public DynamicBackendUtils
    {
    public:
        static bool IsBackendCompatibleTest(const BackendVersion& backendApiVersion,
                                            const BackendVersion& backendVersion)
        {
            return IsBackendCompatibleImpl(backendApiVersion, backendVersion);
        }
    };

    // The backend API version used for the tests
    BackendVersion backendApiVersion{ 2, 4 };

    // Same backend and backend API versions are compatible with the backend API
    BackendVersion sameBackendVersion{ 2, 4 };
    BOOST_TEST(sameBackendVersion == backendApiVersion);
    BOOST_TEST(sameBackendVersion <= backendApiVersion);
    BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, sameBackendVersion) == true);

    // Backend versions that differ from the backend API version by major revision are not compatible
    // with the backend API
    BackendVersion laterMajorBackendVersion{ 3, 4 };
    BOOST_TEST(!(laterMajorBackendVersion == backendApiVersion));
    BOOST_TEST(!(laterMajorBackendVersion <= backendApiVersion));
    BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, laterMajorBackendVersion) == false);

    BackendVersion earlierMajorBackendVersion{ 1, 4 };
    BOOST_TEST(!(earlierMajorBackendVersion == backendApiVersion));
    BOOST_TEST(earlierMajorBackendVersion <= backendApiVersion);
    BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion,
                                                                earlierMajorBackendVersion) == false);

    // Backend versions with the same major revision but later minor revision than
    // the backend API version are not compatible with the backend API
    BackendVersion laterMinorBackendVersion{ 2, 5 };
    BOOST_TEST(!(laterMinorBackendVersion == backendApiVersion));
    BOOST_TEST(!(laterMinorBackendVersion <= backendApiVersion));
    BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, laterMinorBackendVersion) == false);

    // Backend versions with the same major revision but earlier minor revision than
    // the backend API version are compatible with the backend API
    BackendVersion earlierMinorBackendVersion{ 2, 3 };
    BOOST_TEST(!(earlierMinorBackendVersion == backendApiVersion));
    BOOST_TEST(earlierMinorBackendVersion <= backendApiVersion);
    BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, earlierMinorBackendVersion) == true);
}

void CreateValidDynamicBackendObjectTestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Correct backend implementation

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestValidTestDynamicBackendFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_NO_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)));
    BOOST_TEST((dynamicBackend != nullptr));

    BackendId dynamicBackendId;
    BOOST_CHECK_NO_THROW(dynamicBackendId = dynamicBackend->GetBackendId());
    BOOST_TEST((dynamicBackendId == "ValidTestDynamicBackend"));

    BackendVersion dynamicBackendVersion;
    BOOST_CHECK_NO_THROW(dynamicBackendVersion = dynamicBackend->GetBackendVersion());
    BOOST_TEST((dynamicBackendVersion == BackendVersion({ 1, 0 })));

    IBackendInternalUniquePtr dynamicBackendInstance;
    BOOST_CHECK_NO_THROW(dynamicBackendInstance = dynamicBackend->GetBackend());
    BOOST_TEST((dynamicBackendInstance != nullptr));

    BOOST_TEST((dynamicBackendInstance->GetId() == "ValidTestDynamicBackend"));
}

void CreateDynamicBackendObjectInvalidHandleTestImpl()
{
    // Invalid (null) shared object handle

    using namespace armnn;

    void* sharedObjectHandle = nullptr;
    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), InvalidArgumentException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface1TestImpl()
{
    // Valid shared object handle
    // Wrong (not C-style) name mangling

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend1FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface2TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Wrong interface (missing GetBackendId())

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend2FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface3TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Wrong interface (missing GetVersion())

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend3FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface4TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Wrong interface (missing BackendFactory())

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend4FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface5TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Invalid (null) backend id returned by GetBackendId()

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend5FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface6TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Invalid (null) backend instance returned by BackendFactory()

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend6FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_NO_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)));
    BOOST_TEST((dynamicBackend != nullptr));

    BackendId dynamicBackendId;
    BOOST_CHECK_NO_THROW(dynamicBackendId = dynamicBackend->GetBackendId());
    BOOST_TEST((dynamicBackendId == "InvalidTestDynamicBackend"));

    BackendVersion dynamicBackendVersion;
    BOOST_CHECK_NO_THROW(dynamicBackendVersion = dynamicBackend->GetBackendVersion());
    BOOST_TEST((dynamicBackendVersion == BackendVersion({ 1, 0 })));

    IBackendInternalUniquePtr dynamicBackendInstance;
    BOOST_CHECK_THROW(dynamicBackendInstance = dynamicBackend->GetBackend(), RuntimeException);
    BOOST_TEST((dynamicBackendInstance == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface7TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Invalid (incompatible backend API version) backend instance returned by BackendFactory()

    using namespace armnn;

    std::string sharedObjectFilePath = GetTestFilePath(g_TestInvalidTestDynamicBackend7FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}
