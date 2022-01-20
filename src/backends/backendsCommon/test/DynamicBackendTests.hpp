//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendRegistry.hpp>
#include <armnn/backends/DynamicBackend.hpp>
#include <armnn/backends/ILayerSupport.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>
#include <backendsCommon/DynamicBackendUtils.hpp>
#include <armnn/backends/TensorHandle.hpp>
#include <armnnUtils/Filesystem.hpp>
#include <reference/workloads/RefConvolution2dWorkload.hpp>
#include <Runtime.hpp>

#include <string>
#include <memory>

#include <doctest/doctest.h>

#if defined(_MSC_VER)
#include <Windows.h>
#endif

#if !defined(DYNAMIC_BACKEND_BUILD_DIR)
#define DYNAMIC_BACKEND_BUILD_DIR fs::path("./")
#endif

static std::string g_TestDirCLI                             = "--dynamic-backend-build-dir";
static std::string g_TestBaseDir                            = "src/backends/backendsCommon/test/";

static std::string g_TestSharedObjectSubDir                 = "testSharedObject/";
static std::string g_TestDynamicBackendSubDir               = "testDynamicBackend/";

static std::string g_TestSharedObjectFileName               = "libTestSharedObject.so";
static std::string g_TestNoSharedObjectFileName             = "libNoSharedObject.txt";

static std::string g_TestValidTestDynamicBackendFileName    = "libValidTestDynamicBackend.so";
static std::string g_TestInvalidTestDynamicBackend1FileName = "libInvalidTestDynamicBackend1.so";
static std::string g_TestInvalidTestDynamicBackend2FileName = "libInvalidTestDynamicBackend2.so";
static std::string g_TestInvalidTestDynamicBackend3FileName = "libInvalidTestDynamicBackend3.so";
static std::string g_TestInvalidTestDynamicBackend4FileName = "libInvalidTestDynamicBackend4.so";
static std::string g_TestInvalidTestDynamicBackend5FileName = "libInvalidTestDynamicBackend5.so";
static std::string g_TestInvalidTestDynamicBackend6FileName = "libInvalidTestDynamicBackend6.so";
static std::string g_TestInvalidTestDynamicBackend7FileName = "libInvalidTestDynamicBackend7.so";

static std::string g_TestValidBackend2FileName              = "Arm_TestValid2_backend.so";
static std::string g_TestValidBackend3FileName              = "Arm_TestValid3_backend.so";
static std::string g_TestValidBackend4FileName              = "Arm_TestValid4_backend.so";
static std::string g_TestValidBackend5FileName              = "Arm_TestValid5_backend.so";
static std::string g_TestInvalidBackend8FileName            = "Arm_TestInvalid8_backend.so";
static std::string g_TestInvalidBackend9FileName            = "Arm_TestInvalid9_backend.so";
static std::string g_TestInvalidBackend10FileName           = "Arm_TestInvalid10_backend.so";
static std::string g_TestInvalidBackend11FileName           = "Arm_TestInvalid11_backend.so";

static std::string g_TestDynamicBackendsSubDir1  = "backendsTestPath1/";
static std::string g_TestDynamicBackendsSubDir2  = "backendsTestPath2/";
static std::string g_TestDynamicBackendsSubDir3  = "backendsTestPath3/";
static std::string g_TestDynamicBackendsSubDir4  = "backendsTestPath4/";
static std::string g_TestDynamicBackendsSubDir5  = "backendsTestPath5/";
static std::string g_TestDynamicBackendsSubDir6  = "backendsTestPath6/";
static std::string g_TestDynamicBackendsSubDir7  = "backendsTestPath7/";
static std::string g_TestDynamicBackendsSubDir8  = "backendsTestPath8/";
static std::string g_TestDynamicBackendsSubDir9  = "backendsTestPath9/";

static std::string g_DynamicBackendsBaseDir                 = "src/backends/dynamic";
static std::string g_ReferenceDynamicBackendSubDir          = "reference/";
static std::string g_ReferenceBackendFileName               = "Arm_CpuRef_backend.so";

// DynamicBackendUtils wrapper class used for testing (allows to directly invoke the protected methods)
class TestDynamicBackendUtils : public armnn::DynamicBackendUtils
{
public:
    static bool IsBackendCompatibleTest(const armnn::BackendVersion& backendApiVersion,
                                        const armnn::BackendVersion& backendVersion)
    {
        return IsBackendCompatibleImpl(backendApiVersion, backendVersion);
    }

    static std::vector<std::string> GetBackendPathsImplTest(const std::string& path)
    {
        return GetBackendPathsImpl(path);
    }

    static armnn::BackendIdSet RegisterDynamicBackendsImplTest(
            armnn::BackendRegistry& backendRegistry,
            const std::vector<armnn::DynamicBackendPtr>& dynamicBackends)
    {
        return RegisterDynamicBackendsImpl(backendRegistry, dynamicBackends);
    }
};

// BackendRegistry wrapper class used for testing (swaps the underlying factory storage)
class TestBackendRegistry : public armnn::BackendRegistry
{
public:
    TestBackendRegistry() : armnn::BackendRegistry()
    {
        Swap(armnn::BackendRegistryInstance(), m_TempStorage);
    }

    ~TestBackendRegistry()
    {
        Swap(armnn::BackendRegistryInstance(), m_TempStorage);
    }

private:
    FactoryStorage m_TempStorage;
};

#if defined(_MSC_VER)
std::string GetUnitTestExecutablePath()
{
    char buffer[MAX_PATH] = "";
    GetModuleFileNameA(NULL, buffer, MAX_PATH);
    fs::path executablePath(buffer);
    return executablePath.parent_path();
}

#else
std::string GetUnitTestExecutablePath()
{
    char buffer[PATH_MAX] = "";
    if (readlink("/proc/self/exe", buffer, PATH_MAX) != -1)
    {
        fs::path executablePath(buffer);
        return executablePath.parent_path();
    }
    return "";
}
#endif

std::string GetBasePath(const std::string& basePath)
{
    using namespace fs;
    // What we're looking for here is the location of the UnitTests executable.
    // Fall back value of current directory.
    path programLocation = GetUnitTestExecutablePath();
    if (!exists(programLocation))
    {
        programLocation = DYNAMIC_BACKEND_BUILD_DIR;
    }

    // This is the base path from the build where the test libraries were built.
    path sharedObjectPath = programLocation.append(basePath);
    REQUIRE_MESSAGE(exists(sharedObjectPath),
                    ("Base path for shared objects does not exist: " + sharedObjectPath.string()));
    return sharedObjectPath.string();
}

std::string GetTestDirectoryBasePath()
{
    return GetBasePath(g_TestBaseDir);
}

std::string GetDynamicBackendsBasePath()
{
    return GetBasePath(g_DynamicBackendsBaseDir);
}

std::string GetTestSubDirectory(const std::string& subdir)
{
    using namespace fs;

    std::string testDynamicBackendsBaseDir = GetTestDirectoryBasePath();
    path testDynamicBackendsBasePath(testDynamicBackendsBaseDir);
    path testDynamicBackendsSubDir = testDynamicBackendsBasePath.append(subdir);
    // Do not check that the sub-directory exists because for testing reasons we may use non-existing paths

    return testDynamicBackendsSubDir.string();
}

std::string GetTestSubDirectory(const std::string& basePath, const std::string& subdir)
{
    using namespace fs;

    path testDynamicBackendsBasePath(basePath);
    path testDynamicBackendsSubDir = testDynamicBackendsBasePath.append(subdir);
    // Do not check that the sub-directory exists because for testing reasons we may use non-existing paths

    return testDynamicBackendsSubDir.string();
}

std::string GetTestFilePath(const std::string& directory, const std::string& fileName)
{
    using namespace fs;

    path directoryPath(directory);
    path fileNamePath = directoryPath.append(fileName);
    CHECK(exists(fileNamePath));

    return fileNamePath.string();
}

void OpenCloseHandleTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

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
    CHECK_THROWS_AS(sharedObjectHandle = DynamicBackendUtils::OpenHandle(""), RuntimeException);
    CHECK((sharedObjectHandle == nullptr));
}

void OpenNotExistingFileTestImpl()
{
    using namespace armnn;

    void* sharedObjectHandle = nullptr;
    CHECK_THROWS_AS(sharedObjectHandle = DynamicBackendUtils::OpenHandle("NotExistingFileName"), RuntimeException);
    CHECK((sharedObjectHandle == nullptr));
}

void OpenNotSharedObjectTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string notSharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestNoSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_THROWS_AS(sharedObjectHandle = DynamicBackendUtils::OpenHandle(notSharedObjectFilePath), RuntimeException);
    CHECK((sharedObjectHandle == nullptr));
}

void GetValidEntryPointTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    CHECK_NOTHROW(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                    "TestFunction1"));
    CHECK((testFunctionPointer != nullptr));
    CHECK(testFunctionPointer(7) == 7);

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void GetNameMangledEntryPointTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    CHECK_THROWS_AS(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                 "TestFunction2"),
                      RuntimeException);
    CHECK((testFunctionPointer == nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void GetNoExternEntryPointTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    CHECK_THROWS_AS(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                 "TestFunction3"),
                      RuntimeException);
    CHECK((testFunctionPointer == nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void GetNotExistingEntryPointTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    using TestFunctionType = int(*)(int);
    TestFunctionType testFunctionPointer = nullptr;
    CHECK_THROWS_AS(testFunctionPointer = DynamicBackendUtils::GetEntryPoint<TestFunctionType>(sharedObjectHandle,
                                                                                                 "TestFunction4"),
                      RuntimeException);
    CHECK((testFunctionPointer == nullptr));

    DynamicBackendUtils::CloseHandle(sharedObjectHandle);
}

void BackendVersioningTestImpl()
{
    using namespace armnn;

    // The backend API version used for the tests
    BackendVersion backendApiVersion{ 2, 4 };

    // Same backend and backend API versions are compatible with the backend API
    BackendVersion sameBackendVersion{ 2, 4 };
    CHECK(sameBackendVersion == backendApiVersion);
    CHECK(sameBackendVersion <= backendApiVersion);
    CHECK(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, sameBackendVersion) == true);

    // Backend versions that differ from the backend API version by major revision are not compatible
    // with the backend API
    BackendVersion laterMajorBackendVersion{ 3, 4 };
    CHECK(!(laterMajorBackendVersion == backendApiVersion));
    CHECK(!(laterMajorBackendVersion <= backendApiVersion));
    CHECK(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, laterMajorBackendVersion) == false);

    BackendVersion earlierMajorBackendVersion{ 1, 4 };
    CHECK(!(earlierMajorBackendVersion == backendApiVersion));
    CHECK(earlierMajorBackendVersion <= backendApiVersion);
    CHECK(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion,
                                                                earlierMajorBackendVersion) == false);

    // Backend versions with the same major revision but later minor revision than
    // the backend API version are not compatible with the backend API
    BackendVersion laterMinorBackendVersion{ 2, 5 };
    CHECK(!(laterMinorBackendVersion == backendApiVersion));
    CHECK(!(laterMinorBackendVersion <= backendApiVersion));
    CHECK(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, laterMinorBackendVersion) == false);

    // Backend versions with the same major revision but earlier minor revision than
    // the backend API version are compatible with the backend API
    BackendVersion earlierMinorBackendVersion{ 2, 3 };
    CHECK(!(earlierMinorBackendVersion == backendApiVersion));
    CHECK(earlierMinorBackendVersion <= backendApiVersion);
    CHECK(TestDynamicBackendUtils::IsBackendCompatibleTest(backendApiVersion, earlierMinorBackendVersion) == true);
}

#if defined(ARMNNREF_ENABLED)
void CreateValidDynamicBackendObjectTestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Correct backend implementation

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);

    // We expect this path to exists so we can load a valid dynamic backend.
    CHECK_MESSAGE(fs::exists(testSubDirectory),
                  ("Base path for shared objects does not exist: " + testSubDirectory));

    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestValidTestDynamicBackendFileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_NOTHROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)));
    CHECK((dynamicBackend != nullptr));

    BackendId dynamicBackendId;
    CHECK_NOTHROW(dynamicBackendId = dynamicBackend->GetBackendId());
    CHECK((dynamicBackendId == "ValidTestDynamicBackend"));

    BackendVersion dynamicBackendVersion;
    CHECK_NOTHROW(dynamicBackendVersion = dynamicBackend->GetBackendVersion());
    CHECK((dynamicBackendVersion == IBackendInternal::GetApiVersion()));

    IBackendInternalUniquePtr dynamicBackendInstance1;
    CHECK_NOTHROW(dynamicBackendInstance1 = dynamicBackend->GetBackend());
    CHECK((dynamicBackendInstance1 != nullptr));

    BackendRegistry::FactoryFunction dynamicBackendFactoryFunction = nullptr;
    CHECK_NOTHROW(dynamicBackendFactoryFunction = dynamicBackend->GetFactoryFunction());
    CHECK((dynamicBackendFactoryFunction != nullptr));

    IBackendInternalUniquePtr dynamicBackendInstance2;
    CHECK_NOTHROW(dynamicBackendInstance2 = dynamicBackendFactoryFunction());
    CHECK((dynamicBackendInstance2 != nullptr));

    CHECK((dynamicBackendInstance1->GetId() == "ValidTestDynamicBackend"));
    CHECK((dynamicBackendInstance2->GetId() == "ValidTestDynamicBackend"));
}
#endif

void CreateDynamicBackendObjectInvalidHandleTestImpl()
{
    // Invalid (null) shared object handle

    using namespace armnn;

    void* sharedObjectHandle = nullptr;
    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), InvalidArgumentException);
    CHECK((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface1TestImpl()
{
    // Valid shared object handle
    // Wrong (not C-style) name mangling

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend1FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    CHECK((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface2TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Wrong interface (missing GetBackendId())

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend2FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    CHECK((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface3TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Wrong interface (missing GetVersion())

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend3FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    CHECK((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface4TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Wrong interface (missing BackendFactory())

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend4FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    CHECK((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface5TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Invalid (null) backend id returned by GetBackendId()

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend5FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    CHECK((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface6TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Invalid (null) backend instance returned by BackendFactory()

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend6FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_NOTHROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)));
    CHECK((dynamicBackend != nullptr));

    BackendId dynamicBackendId;
    CHECK_NOTHROW(dynamicBackendId = dynamicBackend->GetBackendId());
    CHECK((dynamicBackendId == "InvalidTestDynamicBackend"));

    BackendVersion dynamicBackendVersion;
    CHECK_NOTHROW(dynamicBackendVersion = dynamicBackend->GetBackendVersion());
    CHECK((dynamicBackendVersion >= BackendVersion({ 1, 0 })));

    IBackendInternalUniquePtr dynamicBackendInstance1;
    CHECK_THROWS_AS(dynamicBackendInstance1 = dynamicBackend->GetBackend(), RuntimeException);
    CHECK((dynamicBackendInstance1 == nullptr));

    BackendRegistry::FactoryFunction dynamicBackendFactoryFunction = nullptr;
    CHECK_NOTHROW(dynamicBackendFactoryFunction = dynamicBackend->GetFactoryFunction());
    CHECK((dynamicBackendFactoryFunction != nullptr));

    IBackendInternalUniquePtr dynamicBackendInstance2;
    CHECK_THROWS_AS(dynamicBackendInstance2 = dynamicBackendFactoryFunction(), RuntimeException);
    CHECK((dynamicBackendInstance2 == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface7TestImpl()
{
    // Valid shared object handle
    // Correct name mangling
    // Correct interface
    // Invalid (incompatible backend API version) backend instance returned by BackendFactory()

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend7FileName);

    void* sharedObjectHandle = nullptr;
    CHECK_NOTHROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    CHECK((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    CHECK_THROWS_AS(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    CHECK((dynamicBackend == nullptr));
}

void GetBackendPathsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // The test covers four directories:
    // <unit test path>/src/backends/backendsCommon/test/
    //                                                ├─ backendsTestPath1/   -> exists, contains files
    //                                                ├─ backendsTestPath2/   -> exists, contains files
    //                                                ├─ backendsTestPath3/   -> exists, but empty
    //                                                └─ backendsTestPath4/   -> does not exist

    std::string subDir1 = GetTestSubDirectory(g_TestDynamicBackendsSubDir1);
    std::string subDir2 = GetTestSubDirectory(g_TestDynamicBackendsSubDir2);
    std::string subDir3 = GetTestSubDirectory(g_TestDynamicBackendsSubDir3);
    std::string subDir4 = GetTestSubDirectory(g_TestDynamicBackendsSubDir4);

    CHECK(exists(subDir1));
    CHECK(exists(subDir2));
    CHECK(exists(subDir3));
    CHECK(!exists(subDir4));

    // No path
    CHECK(TestDynamicBackendUtils::GetBackendPathsImplTest("").empty());

    // Malformed path
    std::string malformedDir(subDir1 + "/" + subDir1);
    CHECK(TestDynamicBackendUtils::GetBackendPathsImplTest(malformedDir).size()==0);

    // Single valid path
    std::vector<std::string> DynamicBackendPaths2 = TestDynamicBackendUtils::GetBackendPathsImplTest(subDir1);
    CHECK(DynamicBackendPaths2.size() == 1);
    CHECK(DynamicBackendPaths2[0] == subDir1);

    // Multiple equal and valid paths
    std::string multipleEqualDirs(subDir1 + ":" + subDir1);
    std::vector<std::string> DynamicBackendPaths3 = TestDynamicBackendUtils::GetBackendPathsImplTest(multipleEqualDirs);
    CHECK(DynamicBackendPaths3.size() == 1);
    CHECK(DynamicBackendPaths3[0] == subDir1);

    // Multiple empty paths
    CHECK(TestDynamicBackendUtils::GetBackendPathsImplTest(":::").empty());

    // Multiple valid paths
    std::string multipleValidPaths(subDir1 + ":" + subDir2 + ":" + subDir3);
    std::vector<std::string> DynamicBackendPaths5 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(multipleValidPaths);
    CHECK(DynamicBackendPaths5.size() == 3);
    CHECK(DynamicBackendPaths5[0] == subDir1);
    CHECK(DynamicBackendPaths5[1] == subDir2);
    CHECK(DynamicBackendPaths5[2] == subDir3);

    // Valid among empty paths
    std::string validAmongEmptyDirs("::" + subDir1 + ":");
    std::vector<std::string> DynamicBackendPaths6 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(validAmongEmptyDirs);
    CHECK(DynamicBackendPaths6.size() == 1);
    CHECK(DynamicBackendPaths6[0] == subDir1);

    // Invalid among empty paths
    std::string invalidAmongEmptyDirs(":" + subDir4 + "::");
    CHECK(TestDynamicBackendUtils::GetBackendPathsImplTest(invalidAmongEmptyDirs).empty());

    // Valid, invalid and empty paths
    std::string validInvalidEmptyDirs(subDir1 + ":" + subDir4 + ":");
    std::vector<std::string> DynamicBackendPaths8 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(validInvalidEmptyDirs);
    CHECK(DynamicBackendPaths8.size() == 1);
    CHECK(DynamicBackendPaths8[0] == subDir1);

    // Mix of duplicates of valid, invalid and empty paths
    std::string duplicateValidInvalidEmptyDirs(validInvalidEmptyDirs + ":" + validInvalidEmptyDirs + ":" +
                                               subDir2 + ":" + subDir2);
    std::vector<std::string> DynamicBackendPaths9 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(duplicateValidInvalidEmptyDirs);
    CHECK(DynamicBackendPaths9.size() == 2);
    CHECK(DynamicBackendPaths9[0] == subDir1);
    CHECK(DynamicBackendPaths9[1] == subDir2);
}

void GetBackendPathsOverrideTestImpl()
{
    using namespace armnn;
    using namespace fs;

    std::string subDir1 = GetTestSubDirectory(g_TestDynamicBackendsSubDir1);
    std::string subDir4 = GetTestSubDirectory(g_TestDynamicBackendsSubDir4);

    CHECK(exists(subDir1));
    CHECK(!exists(subDir4));

    // Override with valid path
    std::vector<std::string> validResult = DynamicBackendUtils::GetBackendPaths(subDir1);
    CHECK(validResult.size() == 1);
    CHECK(validResult[0] == subDir1);

    // Override with invalid path
    std::vector<std::string> invalidResult = DynamicBackendUtils::GetBackendPaths(subDir4);
    CHECK(invalidResult.empty());
}

void GetSharedObjectsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // The test covers four directories:
    // <unit test path>/src/backends/backendsCommon/test/
    //                                                ├─ backendsTestPath1/   -> exists, contains files
    //                                                ├─ backendsTestPath2/   -> exists, contains files
    //                                                ├─ backendsTestPath3/   -> exists, but empty
    //                                                └─ backendsTestPath4/   -> does not exist
    //
    // The test sub-directory backendsTestPath1/ contains the following test files:
    //
    // Arm_GpuAcc_backend.so                                       -> valid (basic backend name)
    // Arm_GpuAcc_backend.so.1                                     -> valid (single field version number)
    // Arm_GpuAcc_backend.so.1.2                                   -> valid (multiple field version number)
    // Arm_GpuAcc_backend.so.1.2.3                                 -> valid (multiple field version number)
    // Arm_GpuAcc_backend.so.10.1.27                               -> valid (Multiple digit version)
    // Arm_GpuAcc_backend.so.10.1.33.                              -> not valid (dot not followed by version number)
    // Arm_GpuAcc_backend.so.3.4..5                                -> not valid (dot not followed by version number)
    // Arm_GpuAcc_backend.so.1,1.1                                 -> not valid (comma instead of dot in the version)
    //
    // Arm123_GpuAcc_backend.so                                    -> valid (digits in vendor name are allowed)
    // Arm_GpuAcc456_backend.so                                    -> valid (digits in backend id are allowed)
    // Arm%Co_GpuAcc_backend.so                                    -> not valid (invalid character in vendor name)
    // Arm_Gpu.Acc_backend.so                                      -> not valid (invalid character in backend id)
    //
    // GpuAcc_backend.so                                           -> not valid (missing vendor name)
    // _GpuAcc_backend.so                                          -> not valid (missing vendor name)
    // Arm__backend.so                                             -> not valid (missing backend id)
    // Arm_GpuAcc.so                                               -> not valid (missing "backend" at the end)
    // __backend.so                                                -> not valid (missing vendor name and backend id)
    // __.so                                                       -> not valid (missing all fields)
    //
    // Arm_GpuAcc_backend                                          -> not valid (missing at least ".so" at the end)
    // Arm_GpuAcc_backend_v1.2.so                                  -> not valid (extra version info at the end)
    //
    // The test sub-directory backendsTestPath1/ contains the following test files:
    //
    // Arm_CpuAcc_backend.so                                       -> valid (basic backend name)
    // Arm_CpuAcc_backend.so.1 -> Arm_CpuAcc_backend.so            -> valid (symlink to valid backend file)
    // Arm_CpuAcc_backend.so.1.2 -> Arm_CpuAcc_backend.so.1        -> valid (symlink to valid symlink)
    // Arm_CpuAcc_backend.so.1.2.3 -> Arm_CpuAcc_backend.so.1.2    -> valid (symlink to valid symlink)
    //
    // Arm_no_backend.so -> nothing                                -> not valid (symlink resolves to non-existent file)
    //
    // Arm_GpuAcc_backend.so                                       -> valid (but duplicated from backendsTestPath1/)

    std::string testDynamicBackendsSubDir1 = GetTestSubDirectory(g_TestDynamicBackendsSubDir1);
    std::string testDynamicBackendsSubDir2 = GetTestSubDirectory(g_TestDynamicBackendsSubDir2);
    std::string testDynamicBackendsSubDir3 = GetTestSubDirectory(g_TestDynamicBackendsSubDir3);
    std::string testDynamicBackendsSubDir4 = GetTestSubDirectory(g_TestDynamicBackendsSubDir4);
    CHECK(exists(testDynamicBackendsSubDir1));
    CHECK(exists(testDynamicBackendsSubDir2));
    CHECK(exists(testDynamicBackendsSubDir3));
    CHECK(!exists(testDynamicBackendsSubDir4));

    std::vector<std::string> backendPaths
    {
        testDynamicBackendsSubDir1,
        testDynamicBackendsSubDir2,
        testDynamicBackendsSubDir3,
        testDynamicBackendsSubDir4
    };
    std::vector<std::string> sharedObjects = DynamicBackendUtils::GetSharedObjects(backendPaths);
    std::vector<fs::path> expectedSharedObjects
    {
        path(testDynamicBackendsSubDir1 + "Arm123_GpuAcc_backend.so"),      // Digits in vendor name are allowed
        path(testDynamicBackendsSubDir1 + "Arm_GpuAcc456_backend.so"),      // Digits in backend id are allowed
        path(testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so"),         // Basic backend name
        path(testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.1"),       // Single field version number
        path(testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.1.2"),     // Multiple field version number
        path(testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.1.2.3"),   // Multiple field version number
        path(testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.10.1.27"), // Multiple digit version
        path(testDynamicBackendsSubDir2 + "Arm_CpuAcc_backend.so"),         // Duplicate symlinks removed
        path(testDynamicBackendsSubDir2 + "Arm_GpuAcc_backend.so")          // Duplicates on different paths are allowed
    };

    CHECK(sharedObjects.size() == expectedSharedObjects.size());
    CHECK(fs::equivalent(path(sharedObjects[0]), expectedSharedObjects[0]));
    CHECK(fs::equivalent(path(sharedObjects[1]), expectedSharedObjects[1]));
    CHECK(fs::equivalent(path(sharedObjects[2]), expectedSharedObjects[2]));
    CHECK(fs::equivalent(path(sharedObjects[3]), expectedSharedObjects[3]));
    CHECK(fs::equivalent(path(sharedObjects[4]), expectedSharedObjects[4]));
    CHECK(fs::equivalent(path(sharedObjects[5]), expectedSharedObjects[5]));
    CHECK(fs::equivalent(path(sharedObjects[6]), expectedSharedObjects[6]));
    CHECK(fs::equivalent(path(sharedObjects[7]), expectedSharedObjects[7]));
    CHECK(fs::equivalent(path(sharedObjects[8]), expectedSharedObjects[8]));
}

void CreateDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // The test covers four directories:
    // <unit test path>/src/backends/backendsCommon/test/
    //                                                ├─ backendsTestPath5/   -> exists, contains files
    //                                                ├─ backendsTestPath6/   -> exists, contains files
    //                                                ├─ backendsTestPath7/   -> exists, but empty
    //                                                └─ backendsTestPath8/   -> does not exist
    //
    // The test sub-directory backendsTestPath5/ contains the following test files:
    //
    // Arm_TestValid2_backend.so   -> valid (basic backend name)
    // Arm_TestValid3_backend.so   -> valid (basic backend name)
    // Arm_TestInvalid8_backend.so -> not valid (invalid backend id)
    //
    // The test sub-directory backendsTestPath6/ contains the following test files:
    //
    // Arm_TestValid2_backend.so   -> valid (but duplicated from backendsTestPath5/)
    // Arm_TestValid4_backend.so   -> valid (it has a different filename,
    //                                       but it has the same backend id of Arm_TestValid2_backend.so
    //                                       and the same version)
    // Arm_TestValid5_backend.so   -> valid (basic backend name)
    // Arm_TestInvalid9_backend.so -> not valid (it has a different filename,
    //                                           but it has the same backend id of Arm_TestValid2_backend.so
    //                                           and a version incompatible with the Backend API)

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    std::string testDynamicBackendsSubDir7 = GetTestSubDirectory(g_TestDynamicBackendsSubDir7);
    std::string testDynamicBackendsSubDir8 = GetTestSubDirectory(g_TestDynamicBackendsSubDir8);
    CHECK(exists(testDynamicBackendsSubDir5));
    CHECK(exists(testDynamicBackendsSubDir6));
    CHECK(exists(testDynamicBackendsSubDir7));
    CHECK(!exists(testDynamicBackendsSubDir8));

    std::vector<std::string> backendPaths
    {
        testDynamicBackendsSubDir5,
        testDynamicBackendsSubDir6,
        testDynamicBackendsSubDir7,
        testDynamicBackendsSubDir8
    };
    std::vector<std::string> sharedObjects = DynamicBackendUtils::GetSharedObjects(backendPaths);
    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.size() == 5);
    CHECK((dynamicBackends[0] != nullptr));
    CHECK((dynamicBackends[1] != nullptr));
    CHECK((dynamicBackends[2] != nullptr));
    CHECK((dynamicBackends[3] != nullptr));
    CHECK((dynamicBackends[4] != nullptr));

    // Duplicates are allowed here, they will be skipped later during the backend registration
    CHECK((dynamicBackends[0]->GetBackendId() == "TestValid2"));
    CHECK((dynamicBackends[1]->GetBackendId() == "TestValid3"));
    CHECK((dynamicBackends[2]->GetBackendId() == "TestValid2")); // From duplicate Arm_TestValid2_backend.so
    CHECK((dynamicBackends[3]->GetBackendId() == "TestValid2")); // From Arm_TestValid4_backend.so
    CHECK((dynamicBackends[4]->GetBackendId() == "TestValid5"));
}

void CreateDynamicBackendsNoPathsTestImpl()
{
    using namespace armnn;

    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends({});

    CHECK(dynamicBackends.empty());
}

void CreateDynamicBackendsAllInvalidTestImpl()
{
    using namespace armnn;

    std::vector<std::string> sharedObjects
    {
        "InvalidSharedObject1",
        "InvalidSharedObject2",
        "InvalidSharedObject3",
    };
    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.empty());
}

void CreateDynamicBackendsMixedTypesTestImpl()
{
    using namespace armnn;
    using namespace fs;

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    CHECK(exists(testDynamicBackendsSubDir5));
    CHECK(exists(testDynamicBackendsSubDir6));

    std::string testValidBackend2FilePath = GetTestFilePath(testDynamicBackendsSubDir5,
                                                            g_TestValidBackend2FileName);
    std::string testInvalidBackend8FilePath = GetTestFilePath(testDynamicBackendsSubDir5,
                                                              g_TestInvalidBackend8FileName);
    std::string testInvalidBackend9FilePath = GetTestFilePath(testDynamicBackendsSubDir6,
                                                              g_TestInvalidBackend9FileName);
    CHECK(exists(testValidBackend2FilePath));
    CHECK(exists(testInvalidBackend8FilePath));
    CHECK(exists(testInvalidBackend9FilePath));

    std::vector<std::string> sharedObjects
    {
        testValidBackend2FilePath,   // Arm_TestValid2_backend.so     -> valid (basic backend name)
        testInvalidBackend8FilePath, // Arm_TestInvalid8_backend.so   -> not valid (invalid backend id)
        testInvalidBackend9FilePath, // Arm_TestInvalid9_backend.so   -> not valid (incompatible version)
        "InvalidSharedObject",       // The file does not exist
    };
    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.size() == 1);
    CHECK((dynamicBackends[0] != nullptr));
    CHECK((dynamicBackends[0]->GetBackendId() == "TestValid2"));
}

#if defined(ARMNNREF_ENABLED)
void RegisterSingleDynamicBackendTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Register one valid dynamic backend

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    CHECK(backendRegistry.Size() == 0);

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    CHECK(exists(testDynamicBackendsSubDir5));

    std::string testValidBackend2FilePath = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend2FileName);
    CHECK(exists(testValidBackend2FilePath));

    std::vector<std::string> sharedObjects{ testValidBackend2FilePath };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.size() == 1);
    CHECK((dynamicBackends[0] != nullptr));

    BackendId dynamicBackendId = dynamicBackends[0]->GetBackendId();
    CHECK((dynamicBackendId == "TestValid2"));

    BackendVersion dynamicBackendVersion = dynamicBackends[0]->GetBackendVersion();
    CHECK(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));

    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    CHECK(backendRegistry.Size() == 1);
    CHECK(registeredBackendIds.size() == 1);

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    CHECK(backendIds.size() == 1);
    CHECK((backendIds.find(dynamicBackendId) != backendIds.end()));
    CHECK((registeredBackendIds.find(dynamicBackendId) != registeredBackendIds.end()));

    auto dynamicBackendFactoryFunction = backendRegistry.GetFactory(dynamicBackendId);
    CHECK((dynamicBackendFactoryFunction != nullptr));

    IBackendInternalUniquePtr dynamicBackend = dynamicBackendFactoryFunction();
    CHECK((dynamicBackend != nullptr));
    CHECK((dynamicBackend->GetId() == dynamicBackendId));
}

void RegisterMultipleDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Register many valid dynamic backends

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    CHECK(exists(testDynamicBackendsSubDir5));
    CHECK(exists(testDynamicBackendsSubDir6));

    std::string testValidBackend2FilePath = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend2FileName);
    std::string testValidBackend3FilePath = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend3FileName);
    std::string testValidBackend5FilePath = GetTestFilePath(testDynamicBackendsSubDir6, g_TestValidBackend5FileName);
    CHECK(exists(testValidBackend2FilePath));
    CHECK(exists(testValidBackend3FilePath));
    CHECK(exists(testValidBackend5FilePath));

    std::vector<std::string> sharedObjects
    {
        testValidBackend2FilePath,
        testValidBackend3FilePath,
        testValidBackend5FilePath
    };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.size() == 3);
    CHECK((dynamicBackends[0] != nullptr));
    CHECK((dynamicBackends[1] != nullptr));
    CHECK((dynamicBackends[2] != nullptr));

    BackendId dynamicBackendId1 = dynamicBackends[0]->GetBackendId();
    BackendId dynamicBackendId2 = dynamicBackends[1]->GetBackendId();
    BackendId dynamicBackendId3 = dynamicBackends[2]->GetBackendId();
    CHECK((dynamicBackendId1 == "TestValid2"));
    CHECK((dynamicBackendId2 == "TestValid3"));
    CHECK((dynamicBackendId3 == "TestValid5"));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendVersion dynamicBackendVersion = dynamicBackends[i]->GetBackendVersion();
        CHECK(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));
    }

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    CHECK(backendRegistry.Size() == 0);

    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    CHECK(backendRegistry.Size() == 3);
    CHECK(registeredBackendIds.size() == 3);

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    CHECK(backendIds.size() == 3);
    CHECK((backendIds.find(dynamicBackendId1) != backendIds.end()));
    CHECK((backendIds.find(dynamicBackendId2) != backendIds.end()));
    CHECK((backendIds.find(dynamicBackendId3) != backendIds.end()));
    CHECK((registeredBackendIds.find(dynamicBackendId1) != registeredBackendIds.end()));
    CHECK((registeredBackendIds.find(dynamicBackendId2) != registeredBackendIds.end()));
    CHECK((registeredBackendIds.find(dynamicBackendId3) != registeredBackendIds.end()));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendId dynamicBackendId = dynamicBackends[i]->GetBackendId();

        auto dynamicBackendFactoryFunction = backendRegistry.GetFactory(dynamicBackendId);
        CHECK((dynamicBackendFactoryFunction != nullptr));

        IBackendInternalUniquePtr dynamicBackend = dynamicBackendFactoryFunction();
        CHECK((dynamicBackend != nullptr));
        CHECK((dynamicBackend->GetId() == dynamicBackendId));
    }
}

void RegisterMixedDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // The test covers five directories:
    // <unit test path>/src/backends/backendsCommon/test/
    //                                                ├─ backendsTestPath5/   -> exists, contains files
    //                                                ├─ backendsTestPath6/   -> exists, contains files
    //                                                ├─ backendsTestPath7/   -> exists, but empty
    //                                                ├─ backendsTestPath8/   -> does not exist
    //                                                └─ backendsTestPath9/   -> exists, contains files
    //
    // The test sub-directory backendsTestPath5/ contains the following test files:
    //
    // Arm_TestValid2_backend.so   -> valid (basic backend name)
    // Arm_TestValid3_backend.so   -> valid (basic backend name)
    // Arm_TestInvalid8_backend.so -> not valid (invalid backend id)
    //
    // The test sub-directory backendsTestPath6/ contains the following test files:
    //
    // Arm_TestValid2_backend.so   -> valid (but duplicated from backendsTestPath5/)
    // Arm_TestValid4_backend.so   -> valid (it has a different filename,
    //                                       but it has the same backend id of Arm_TestValid2_backend.so
    //                                       and the same version)
    // Arm_TestValid5_backend.so   -> valid (basic backend name)
    // Arm_TestInvalid9_backend.so -> not valid (it has a different filename,
    //                                           but it has the same backend id of Arm_TestValid2_backend.so
    //                                           and a version incompatible with the Backend API)
    //
    // The test sub-directory backendsTestPath9/ contains the following test files:
    //
    // Arm_TestInvalid10_backend.so -> not valid (empty backend id)
    // Arm_TestInvalid11_backend.so -> not valid ("Unknown" backend id)

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    std::string testDynamicBackendsSubDir7 = GetTestSubDirectory(g_TestDynamicBackendsSubDir7);
    std::string testDynamicBackendsSubDir8 = GetTestSubDirectory(g_TestDynamicBackendsSubDir8);
    std::string testDynamicBackendsSubDir9 = GetTestSubDirectory(g_TestDynamicBackendsSubDir9);
    CHECK(exists(testDynamicBackendsSubDir5));
    CHECK(exists(testDynamicBackendsSubDir6));
    CHECK(exists(testDynamicBackendsSubDir7));
    CHECK(!exists(testDynamicBackendsSubDir8));
    CHECK(exists(testDynamicBackendsSubDir9));

    std::string testValidBackend2FilePath    = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend2FileName);
    std::string testValidBackend3FilePath    = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend3FileName);
    std::string testValidBackend2DupFilePath = GetTestFilePath(testDynamicBackendsSubDir6, g_TestValidBackend2FileName);
    std::string testValidBackend4FilePath    = GetTestFilePath(testDynamicBackendsSubDir6, g_TestValidBackend4FileName);
    std::string testValidBackend5FilePath    = GetTestFilePath(testDynamicBackendsSubDir6, g_TestValidBackend5FileName);
    std::string testInvalidBackend8FilePath  = GetTestFilePath(testDynamicBackendsSubDir5,
                                                               g_TestInvalidBackend8FileName);
    std::string testInvalidBackend9FilePath  = GetTestFilePath(testDynamicBackendsSubDir6,
                                                               g_TestInvalidBackend9FileName);
    std::string testInvalidBackend10FilePath = GetTestFilePath(testDynamicBackendsSubDir9,
                                                               g_TestInvalidBackend10FileName);
    std::string testInvalidBackend11FilePath = GetTestFilePath(testDynamicBackendsSubDir9,
                                                               g_TestInvalidBackend11FileName);
    CHECK(exists(testValidBackend2FilePath));
    CHECK(exists(testValidBackend3FilePath));
    CHECK(exists(testValidBackend2DupFilePath));
    CHECK(exists(testValidBackend4FilePath));
    CHECK(exists(testValidBackend5FilePath));
    CHECK(exists(testInvalidBackend8FilePath));
    CHECK(exists(testInvalidBackend9FilePath));
    CHECK(exists(testInvalidBackend10FilePath));
    CHECK(exists(testInvalidBackend11FilePath));

    std::vector<std::string> sharedObjects
    {
        testValidBackend2FilePath,
        testValidBackend3FilePath,
        testValidBackend2DupFilePath,
        testValidBackend4FilePath,
        testValidBackend5FilePath,
        testInvalidBackend8FilePath,
        testInvalidBackend9FilePath,
        testInvalidBackend10FilePath,
        testInvalidBackend11FilePath,
        "InvalidSharedObject"
    };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.size() == 7);
    CHECK((dynamicBackends[0] != nullptr));
    CHECK((dynamicBackends[1] != nullptr));
    CHECK((dynamicBackends[2] != nullptr));
    CHECK((dynamicBackends[3] != nullptr));
    CHECK((dynamicBackends[4] != nullptr));
    CHECK((dynamicBackends[5] != nullptr));
    CHECK((dynamicBackends[6] != nullptr));

    BackendId dynamicBackendId1 = dynamicBackends[0]->GetBackendId();
    BackendId dynamicBackendId2 = dynamicBackends[1]->GetBackendId();
    BackendId dynamicBackendId3 = dynamicBackends[2]->GetBackendId();
    BackendId dynamicBackendId4 = dynamicBackends[3]->GetBackendId();
    BackendId dynamicBackendId5 = dynamicBackends[4]->GetBackendId();
    BackendId dynamicBackendId6 = dynamicBackends[5]->GetBackendId();
    BackendId dynamicBackendId7 = dynamicBackends[6]->GetBackendId();
    CHECK((dynamicBackendId1 == "TestValid2"));
    CHECK((dynamicBackendId2 == "TestValid3"));
    CHECK((dynamicBackendId3 == "TestValid2")); // From duplicate Arm_TestValid2_backend.so
    CHECK((dynamicBackendId4 == "TestValid2")); // From Arm_TestValid4_backend.so
    CHECK((dynamicBackendId5 == "TestValid5"));
    CHECK((dynamicBackendId6 == ""));
    CHECK((dynamicBackendId7 == "Unknown"));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendVersion dynamicBackendVersion = dynamicBackends[i]->GetBackendVersion();
        CHECK(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));
    }

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    CHECK(backendRegistry.Size() == 0);

    std::vector<BackendId> expectedRegisteredbackendIds
    {
        "TestValid2",
        "TestValid3",
        "TestValid5"
    };

    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    CHECK(backendRegistry.Size() == expectedRegisteredbackendIds.size());
    CHECK(registeredBackendIds.size() == expectedRegisteredbackendIds.size());

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    CHECK(backendIds.size() == expectedRegisteredbackendIds.size());
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        CHECK((backendIds.find(expectedRegisteredbackendId) != backendIds.end()));
        CHECK((registeredBackendIds.find(expectedRegisteredbackendId) != registeredBackendIds.end()));

        auto dynamicBackendFactoryFunction = backendRegistry.GetFactory(expectedRegisteredbackendId);
        CHECK((dynamicBackendFactoryFunction != nullptr));

        IBackendInternalUniquePtr dynamicBackend = dynamicBackendFactoryFunction();
        CHECK((dynamicBackend != nullptr));
        CHECK((dynamicBackend->GetId() == expectedRegisteredbackendId));
    }
}
#endif

void RegisterMultipleInvalidDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Try to register many invalid dynamic backends

    // The test covers one directory:
    // <unit test path>/src/backends/backendsCommon/test/
    //                                                └─ backendsTestPath9/   -> exists, contains files
    //
    // The test sub-directory backendsTestPath9/ contains the following test files:
    //
    // Arm_TestInvalid10_backend.so -> not valid (invalid backend id)
    // Arm_TestInvalid11_backend.so -> not valid (invalid backend id)

    std::string testDynamicBackendsSubDir9 = GetTestSubDirectory(g_TestDynamicBackendsSubDir9);
    CHECK(exists(testDynamicBackendsSubDir9));

    std::string testInvalidBackend10FilePath = GetTestFilePath(testDynamicBackendsSubDir9,
                                                               g_TestInvalidBackend10FileName);
    std::string testInvalidBackend11FilePath = GetTestFilePath(testDynamicBackendsSubDir9,
                                                               g_TestInvalidBackend11FileName);
    CHECK(exists(testInvalidBackend10FilePath));
    CHECK(exists(testInvalidBackend11FilePath));

    std::vector<std::string> sharedObjects
    {
        testInvalidBackend10FilePath,
        testInvalidBackend11FilePath,
        "InvalidSharedObject"
    };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    CHECK(dynamicBackends.size() == 2);
    CHECK((dynamicBackends[0] != nullptr));
    CHECK((dynamicBackends[1] != nullptr));

    BackendId dynamicBackendId1 = dynamicBackends[0]->GetBackendId();
    BackendId dynamicBackendId2 = dynamicBackends[1]->GetBackendId();
    CHECK((dynamicBackendId1 == ""));
    CHECK((dynamicBackendId2 == "Unknown"));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendVersion dynamicBackendVersion = dynamicBackends[i]->GetBackendVersion();
        CHECK(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));
    }

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    CHECK(backendRegistry.Size() == 0);

    // Check that no dynamic backend got registered
    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    CHECK(backendRegistry.Size() == 0);
    CHECK(registeredBackendIds.empty());
}

#if !defined(ARMNN_DYNAMIC_BACKEND_ENABLED)

void RuntimeEmptyTestImpl()
{
    using namespace armnn;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() == 0);

    IRuntime::CreationOptions creationOptions;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.empty());

    CHECK(backendRegistry.Size() == 0);
}

#endif

void RuntimeDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains valid and invalid backends
    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    CHECK(exists(testDynamicBackendsSubDir5));

    // Using the path override in CreationOptions to load some test dynamic backends
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = testDynamicBackendsSubDir5;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    std::vector<BackendId> expectedRegisteredbackendIds
    {
        "TestValid2",
        "TestValid3"
    };

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() == expectedRegisteredbackendIds.size());

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        CHECK((backendIds.find(expectedRegisteredbackendId) != backendIds.end()));
    }

    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.size() == expectedRegisteredbackendIds.size());
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        CHECK((supportedBackendIds.find(expectedRegisteredbackendId) != supportedBackendIds.end()));
    }
}

void RuntimeDuplicateDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains valid, invalid and duplicate backends
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    CHECK(exists(testDynamicBackendsSubDir6));

    // Using the path override in CreationOptions to load some test dynamic backends
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = testDynamicBackendsSubDir6;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    std::vector<BackendId> expectedRegisteredbackendIds
    {
        "TestValid2",
        "TestValid5"
    };

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() == expectedRegisteredbackendIds.size());

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        CHECK((backendIds.find(expectedRegisteredbackendId) != backendIds.end()));
    }

    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.size() == expectedRegisteredbackendIds.size());
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        CHECK((supportedBackendIds.find(expectedRegisteredbackendId) != supportedBackendIds.end()));
    }
}

void RuntimeInvalidDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains only invalid backends
    std::string testDynamicBackendsSubDir9 = GetTestSubDirectory(g_TestDynamicBackendsSubDir9);
    CHECK(exists(testDynamicBackendsSubDir9));

    // Using the path override in CreationOptions to load some test dynamic backends
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = testDynamicBackendsSubDir9;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() == 0);

    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.empty());
}

void RuntimeInvalidOverridePathTestImpl()
{
    using namespace armnn;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // Using the path override in CreationOptions to load some test dynamic backends
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = "InvalidPath";
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() == 0);

    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.empty());
}

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

void CreateReferenceDynamicBackendTestImpl()
{
    using namespace armnn;
    using namespace fs;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains the reference dynamic backend
    std::string dynamicBackendsBaseDir = GetDynamicBackendsBasePath();
    std::string referenceDynamicBackendSubDir = GetTestSubDirectory(dynamicBackendsBaseDir,
                                                                    g_ReferenceDynamicBackendSubDir);
    CHECK(exists(referenceDynamicBackendSubDir));

    // Check that the reference dynamic backend file exists
    std::string referenceBackendFilePath = GetTestFilePath(referenceDynamicBackendSubDir,
                                                           g_ReferenceBackendFileName);
    CHECK(exists(referenceBackendFilePath));

    // Using the path override in CreationOptions to load the reference dynamic backend
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = referenceDynamicBackendSubDir;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() == 1);

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    CHECK((backendIds.find("CpuRef") != backendIds.end()));

    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.size() == 1);
    CHECK((supportedBackendIds.find("CpuRef") != supportedBackendIds.end()));

    // Get the factory function
    auto referenceDynamicBackendFactoryFunction = backendRegistry.GetFactory("CpuRef");
    CHECK((referenceDynamicBackendFactoryFunction != nullptr));

    // Use the factory function to create an instance of the reference backend
    IBackendInternalUniquePtr referenceDynamicBackend = referenceDynamicBackendFactoryFunction();
    CHECK((referenceDynamicBackend != nullptr));
    CHECK((referenceDynamicBackend->GetId() == "CpuRef"));

    // Test the backend instance by querying the layer support
    IBackendInternal::ILayerSupportSharedPtr referenceLayerSupport = referenceDynamicBackend->GetLayerSupport();
    CHECK((referenceLayerSupport != nullptr));

    TensorShape inputShape {  1, 16, 16, 16 };
    TensorShape outputShape{  1, 16, 16, 16 };
    TensorShape weightShape{ 16,  1,  1, 16 };
    TensorInfo inputInfo (inputShape,  DataType::Float32);
    TensorInfo outputInfo(outputShape, DataType::Float32);
    TensorInfo weightInfo(weightShape, DataType::Float32);
    Convolution2dDescriptor convolution2dDescriptor;
    std::vector<TensorInfo> infos = {inputInfo, outputInfo, weightInfo, TensorInfo()};
    bool referenceConvolution2dSupported =
             referenceLayerSupport->IsLayerSupported(LayerType::Convolution2d,
                                                     infos,
                                                     convolution2dDescriptor);
    CHECK(referenceConvolution2dSupported);

    // Test the backend instance by creating a workload
    IBackendInternal::IWorkloadFactoryPtr referenceWorkloadFactory = referenceDynamicBackend->CreateWorkloadFactory();
    CHECK((referenceWorkloadFactory != nullptr));

    // Create dummy settings for the workload
    Convolution2dQueueDescriptor convolution2dQueueDescriptor;
    WorkloadInfo workloadInfo
    {
        { inputInfo },
        { outputInfo }
    };
    convolution2dQueueDescriptor.m_Inputs.push_back(nullptr);
    auto weights = std::make_unique<ScopedTensorHandle>(weightInfo);
    convolution2dQueueDescriptor.m_Weight = weights.get();

    // Create a convolution workload with the dummy settings
    auto workload = referenceWorkloadFactory->CreateWorkload(LayerType::Convolution2d,
                                                             convolution2dQueueDescriptor,
                                                             workloadInfo);
    CHECK((workload != nullptr));
    CHECK(workload.get() == PolymorphicDowncast<RefConvolution2dWorkload*>(workload.get()));
}

#endif

#if defined(SAMPLE_DYNAMIC_BACKEND_ENABLED)

void CheckSampleDynamicBackendLoaded()
{
    using namespace armnn;
    // At this point we expect DYNAMIC_BACKEND_PATHS to include a path to where libArm_SampleDynamic_backend.so is.
    // If it hasn't been loaded there's no point continuing with the rest of the tests.
    BackendIdSet backendIds = BackendRegistryInstance().GetBackendIds();
    if (backendIds.find("SampleDynamic") == backendIds.end())
    {
        std::string message = "The SampleDynamic backend has not been loaded. This may be a build configuration error. "
                              "Ensure a DYNAMIC_BACKEND_PATHS was set at compile time to the location of "
                              "libArm_SampleDynamic_backend.so. "
                              "To disable this test recompile with: -DSAMPLE_DYNAMIC_BACKEND_ENABLED=0";
        FAIL(message);
    }
}

void CreateSampleDynamicBackendTestImpl()
{
    using namespace armnn;
    // Using the path override in CreationOptions to load the reference dynamic backend
    IRuntime::CreationOptions creationOptions;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);
    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    CHECK(backendRegistry.Size() >= 1);
    CheckSampleDynamicBackendLoaded();
    const DeviceSpec& deviceSpec = *PolymorphicDowncast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    CHECK(supportedBackendIds.size()>= 1);
    CHECK((supportedBackendIds.find("SampleDynamic") != supportedBackendIds.end()));

    // Get the factory function
    auto sampleDynamicBackendFactoryFunction = backendRegistry.GetFactory("SampleDynamic");
    CHECK((sampleDynamicBackendFactoryFunction != nullptr));

    // Use the factory function to create an instance of the dynamic backend
    IBackendInternalUniquePtr sampleDynamicBackend = sampleDynamicBackendFactoryFunction();
    CHECK((sampleDynamicBackend != nullptr));
    CHECK((sampleDynamicBackend->GetId() == "SampleDynamic"));

    // Test the backend instance by querying the layer support
    IBackendInternal::ILayerSupportSharedPtr sampleLayerSupport = sampleDynamicBackend->GetLayerSupport();
    CHECK((sampleLayerSupport != nullptr));

    TensorShape inputShape {  1, 16, 16, 16 };
    TensorShape outputShape{  1, 16, 16, 16 };
    TensorShape weightShape{ 16,  1,  1, 16 };
    TensorInfo inputInfo (inputShape,  DataType::Float32);
    TensorInfo outputInfo(outputShape, DataType::Float32);
    TensorInfo weightInfo(weightShape, DataType::Float32);
    Convolution2dDescriptor convolution2dDescriptor;
    std::vector<TensorInfo> infos = {inputInfo, outputInfo, weightInfo, TensorInfo()};
    bool sampleConvolution2dSupported =
             sampleLayerSupport->IsLayerSupported(LayerType::Convolution2d,
                                                  infos,
                                                  convolution2dDescriptor);
    CHECK(!sampleConvolution2dSupported);

    // Test the backend instance by creating a workload
    IBackendInternal::IWorkloadFactoryPtr sampleWorkloadFactory = sampleDynamicBackend->CreateWorkloadFactory();
    CHECK((sampleWorkloadFactory != nullptr));

    // Create dummy settings for the workload
    AdditionQueueDescriptor additionQueueDescriptor;
    WorkloadInfo workloadInfo
    {
        { inputInfo, inputInfo },
        { outputInfo }
    };

    // Create a addition workload
    auto workload = sampleWorkloadFactory->CreateWorkload(LayerType::Addition, additionQueueDescriptor, workloadInfo);
    CHECK((workload != nullptr));
}

void SampleDynamicBackendEndToEndTestImpl()
{
    using namespace armnn;
    // Create runtime in which test will run
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));
    CheckSampleDynamicBackendLoaded();
    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input0 = net->AddInputLayer(0);
    IConnectableLayer* input1 = net->AddInputLayer(1);
    IConnectableLayer* add = net->AddAdditionLayer();
    IConnectableLayer* output = net->AddOutputLayer(0);

    input0->GetOutputSlot(0).Connect(add->GetInputSlot(0));
    input1->GetOutputSlot(0).Connect(add->GetInputSlot(1));
    add->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    TensorInfo tensorInfo(TensorShape({2, 1}), DataType::Float32);
    input0->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    input1->GetOutputSlot(0).SetTensorInfo(tensorInfo);
    add->GetOutputSlot(0).SetTensorInfo(tensorInfo);

    // optimize the network
    IOptimizedNetworkPtr optNet = Optimize(*net, {"SampleDynamic"}, runtime->GetDeviceSpec());

    // Loads it into the runtime.
    NetworkId netId;
    runtime->LoadNetwork(netId, std::move(optNet));

    std::vector<float> input0Data{ 5.0f, 3.0f };
    std::vector<float> input1Data{ 10.0f, 8.0f };
    std::vector<float> expectedOutputData{ 15.0f, 11.0f };
    std::vector<float> outputData(2);

    TensorInfo inputTensorInfo = runtime->GetInputTensorInfo(netId, 0);
    inputTensorInfo.SetConstant(true);
    InputTensors inputTensors
        {
            {0,armnn::ConstTensor(inputTensorInfo, input0Data.data())},
            {1,armnn::ConstTensor(inputTensorInfo, input1Data.data())}
        };
    OutputTensors outputTensors
        {
            {0,armnn::Tensor(runtime->GetOutputTensorInfo(netId, 0), outputData.data())}
        };

    // Does the inference.
    runtime->EnqueueWorkload(netId, inputTensors, outputTensors);

    // Checks the results.
    CHECK(outputData == expectedOutputData);
}
#endif
