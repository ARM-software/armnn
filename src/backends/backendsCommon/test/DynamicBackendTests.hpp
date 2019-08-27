//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <backendsCommon/DynamicBackend.hpp>
#include <backendsCommon/DynamicBackendUtils.hpp>
#include <backendsCommon/BackendRegistry.hpp>
#include <backendsCommon/CpuTensorHandle.hpp>

#include <armnn/ILayerSupport.hpp>

#include <reference/workloads/RefConvolution2dWorkload.hpp>

#include <Runtime.hpp>

#include <string>
#include <memory>
#include <string>

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/dll.hpp>

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

std::string GetBasePath(const std::string& basePath)
{
    using namespace boost::filesystem;

    path programLocation = boost::dll::program_location().parent_path();
    path sharedObjectPath = programLocation.append(basePath);
    BOOST_CHECK(exists(sharedObjectPath));

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
    using namespace boost::filesystem;

    std::string testDynamicBackendsBaseDir = GetTestDirectoryBasePath();
    path testDynamicBackendsBasePath(testDynamicBackendsBaseDir);
    path testDynamicBackendsSubDir = testDynamicBackendsBasePath.append(subdir);
    // Do not check that the sub-directory exists because for testing reasons we may use non-existing paths

    return testDynamicBackendsSubDir.string();
}

std::string GetTestSubDirectory(const std::string& basePath, const std::string& subdir)
{
    using namespace boost::filesystem;

    path testDynamicBackendsBasePath(basePath);
    path testDynamicBackendsSubDir = testDynamicBackendsBasePath.append(subdir);
    // Do not check that the sub-directory exists because for testing reasons we may use non-existing paths

    return testDynamicBackendsSubDir.string();
}

std::string GetTestFilePath(const std::string& directory, const std::string& fileName)
{
    using namespace boost::filesystem;

    path directoryPath(directory);
    path fileNamePath = directoryPath.append(fileName);
    BOOST_CHECK(exists(fileNamePath));

    return fileNamePath.string();
}

void OpenCloseHandleTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string notSharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestNoSharedObjectFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(notSharedObjectFilePath), RuntimeException);
    BOOST_TEST((sharedObjectHandle == nullptr));
}

void GetValidEntryPointTestImpl()
{
    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestSharedObjectSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestSharedObjectFileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestValidTestDynamicBackendFileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_NO_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)));
    BOOST_TEST((dynamicBackend != nullptr));

    BackendId dynamicBackendId;
    BOOST_CHECK_NO_THROW(dynamicBackendId = dynamicBackend->GetBackendId());
    BOOST_TEST((dynamicBackendId == "ValidTestDynamicBackend"));

    BackendVersion dynamicBackendVersion;
    BOOST_CHECK_NO_THROW(dynamicBackendVersion = dynamicBackend->GetBackendVersion());
    BOOST_TEST((dynamicBackendVersion == IBackendInternal::GetApiVersion()));

    IBackendInternalUniquePtr dynamicBackendInstance1;
    BOOST_CHECK_NO_THROW(dynamicBackendInstance1 = dynamicBackend->GetBackend());
    BOOST_TEST((dynamicBackendInstance1 != nullptr));

    BackendRegistry::FactoryFunction dynamicBackendFactoryFunction = nullptr;
    BOOST_CHECK_NO_THROW(dynamicBackendFactoryFunction = dynamicBackend->GetFactoryFunction());
    BOOST_TEST((dynamicBackendFactoryFunction != nullptr));

    IBackendInternalUniquePtr dynamicBackendInstance2;
    BOOST_CHECK_NO_THROW(dynamicBackendInstance2 = dynamicBackendFactoryFunction());
    BOOST_TEST((dynamicBackendInstance2 != nullptr));

    BOOST_TEST((dynamicBackendInstance1->GetId() == "ValidTestDynamicBackend"));
    BOOST_TEST((dynamicBackendInstance2->GetId() == "ValidTestDynamicBackend"));
}

void CreateDynamicBackendObjectInvalidHandleTestImpl()
{
    // Invalid (null) shared object handle

    using namespace armnn;

    void* sharedObjectHandle = nullptr;
    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), InvalidArgumentException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void CreateDynamicBackendObjectInvalidInterface1TestImpl()
{
    // Valid shared object handle
    // Wrong (not C-style) name mangling

    using namespace armnn;

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend1FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
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
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
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
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
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
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend5FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend6FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_NO_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)));
    BOOST_TEST((dynamicBackend != nullptr));

    BackendId dynamicBackendId;
    BOOST_CHECK_NO_THROW(dynamicBackendId = dynamicBackend->GetBackendId());
    BOOST_TEST((dynamicBackendId == "InvalidTestDynamicBackend"));

    BackendVersion dynamicBackendVersion;
    BOOST_CHECK_NO_THROW(dynamicBackendVersion = dynamicBackend->GetBackendVersion());
    BOOST_TEST((dynamicBackendVersion == BackendVersion({ 1, 0 })));

    IBackendInternalUniquePtr dynamicBackendInstance1;
    BOOST_CHECK_THROW(dynamicBackendInstance1 = dynamicBackend->GetBackend(), RuntimeException);
    BOOST_TEST((dynamicBackendInstance1 == nullptr));

    BackendRegistry::FactoryFunction dynamicBackendFactoryFunction = nullptr;
    BOOST_CHECK_NO_THROW(dynamicBackendFactoryFunction = dynamicBackend->GetFactoryFunction());
    BOOST_TEST((dynamicBackendFactoryFunction != nullptr));

    IBackendInternalUniquePtr dynamicBackendInstance2;
    BOOST_CHECK_THROW(dynamicBackendInstance2 = dynamicBackendFactoryFunction(), RuntimeException);
    BOOST_TEST((dynamicBackendInstance2 == nullptr));
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
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    DynamicBackendPtr dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void GetBackendPathsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

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

    BOOST_CHECK(exists(subDir1));
    BOOST_CHECK(exists(subDir2));
    BOOST_CHECK(exists(subDir3));
    BOOST_CHECK(!exists(subDir4));

    // No path
    BOOST_TEST(TestDynamicBackendUtils::GetBackendPathsImplTest("").empty());

    // Malformed path
    std::string malformedDir(subDir1 + "/" + subDir1);
    BOOST_TEST(TestDynamicBackendUtils::GetBackendPathsImplTest(malformedDir).size()==0);

    // Single valid path
    std::vector<std::string> DynamicBackendPaths2 = TestDynamicBackendUtils::GetBackendPathsImplTest(subDir1);
    BOOST_TEST(DynamicBackendPaths2.size() == 1);
    BOOST_TEST(DynamicBackendPaths2[0] == subDir1);

    // Multiple equal and valid paths
    std::string multipleEqualDirs(subDir1 + ":" + subDir1);
    std::vector<std::string> DynamicBackendPaths3 = TestDynamicBackendUtils::GetBackendPathsImplTest(multipleEqualDirs);
    BOOST_TEST(DynamicBackendPaths3.size() == 1);
    BOOST_TEST(DynamicBackendPaths3[0] == subDir1);

    // Multiple empty paths
    BOOST_TEST(TestDynamicBackendUtils::GetBackendPathsImplTest(":::").empty());

    // Multiple valid paths
    std::string multipleValidPaths(subDir1 + ":" + subDir2 + ":" + subDir3);
    std::vector<std::string> DynamicBackendPaths5 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(multipleValidPaths);
    BOOST_TEST(DynamicBackendPaths5.size() == 3);
    BOOST_TEST(DynamicBackendPaths5[0] == subDir1);
    BOOST_TEST(DynamicBackendPaths5[1] == subDir2);
    BOOST_TEST(DynamicBackendPaths5[2] == subDir3);

    // Valid among empty paths
    std::string validAmongEmptyDirs("::" + subDir1 + ":");
    std::vector<std::string> DynamicBackendPaths6 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(validAmongEmptyDirs);
    BOOST_TEST(DynamicBackendPaths6.size() == 1);
    BOOST_TEST(DynamicBackendPaths6[0] == subDir1);

    // Invalid among empty paths
    std::string invalidAmongEmptyDirs(":" + subDir4 + "::");
    BOOST_TEST(TestDynamicBackendUtils::GetBackendPathsImplTest(invalidAmongEmptyDirs).empty());

    // Valid, invalid and empty paths
    std::string validInvalidEmptyDirs(subDir1 + ":" + subDir4 + ":");
    std::vector<std::string> DynamicBackendPaths8 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(validInvalidEmptyDirs);
    BOOST_TEST(DynamicBackendPaths8.size() == 1);
    BOOST_TEST(DynamicBackendPaths8[0] == subDir1);

    // Mix of duplicates of valid, invalid and empty paths
    std::string duplicateValidInvalidEmptyDirs(validInvalidEmptyDirs + ":" + validInvalidEmptyDirs + ":" +
                                               subDir2 + ":" + subDir2);
    std::vector<std::string> DynamicBackendPaths9 =
        TestDynamicBackendUtils::GetBackendPathsImplTest(duplicateValidInvalidEmptyDirs);
    BOOST_TEST(DynamicBackendPaths9.size() == 2);
    BOOST_TEST(DynamicBackendPaths9[0] == subDir1);
    BOOST_TEST(DynamicBackendPaths9[1] == subDir2);
}

void GetBackendPathsOverrideTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    std::string subDir1 = GetTestSubDirectory(g_TestDynamicBackendsSubDir1);
    std::string subDir4 = GetTestSubDirectory(g_TestDynamicBackendsSubDir4);

    BOOST_CHECK(exists(subDir1));
    BOOST_CHECK(!exists(subDir4));

    // Override with valid path
    std::vector<std::string> validResult = DynamicBackendUtils::GetBackendPaths(subDir1);
    BOOST_TEST(validResult.size() == 1);
    BOOST_TEST(validResult[0] == subDir1);

    // Override with invalid path
    std::vector<std::string> invalidResult = DynamicBackendUtils::GetBackendPaths(subDir4);
    BOOST_TEST(invalidResult.empty());
}

void GetSharedObjectsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

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
    BOOST_CHECK(exists(testDynamicBackendsSubDir1));
    BOOST_CHECK(exists(testDynamicBackendsSubDir2));
    BOOST_CHECK(exists(testDynamicBackendsSubDir3));
    BOOST_CHECK(!exists(testDynamicBackendsSubDir4));

    std::vector<std::string> backendPaths
    {
        testDynamicBackendsSubDir1,
        testDynamicBackendsSubDir2,
        testDynamicBackendsSubDir3,
        testDynamicBackendsSubDir4
    };
    std::vector<std::string> sharedObjects = DynamicBackendUtils::GetSharedObjects(backendPaths);
    std::vector<std::string> expectedSharedObjects
    {
        testDynamicBackendsSubDir1 + "Arm123_GpuAcc_backend.so",      // Digits in vendor name are allowed
        testDynamicBackendsSubDir1 + "Arm_GpuAcc456_backend.so",      // Digits in backend id are allowed
        testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so",         // Basic backend name
        testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.1",       // Single field version number
        testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.1.2",     // Multiple field version number
        testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.1.2.3",   // Multiple field version number
        testDynamicBackendsSubDir1 + "Arm_GpuAcc_backend.so.10.1.27", // Multiple digit version
        testDynamicBackendsSubDir2 + "Arm_CpuAcc_backend.so",         // Duplicate symlinks removed
        testDynamicBackendsSubDir2 + "Arm_GpuAcc_backend.so"          // Duplicates on different paths are allowed
    };

    BOOST_TEST(sharedObjects.size() == expectedSharedObjects.size());
    BOOST_TEST(sharedObjects[0] == expectedSharedObjects[0]);
    BOOST_TEST(sharedObjects[1] == expectedSharedObjects[1]);
    BOOST_TEST(sharedObjects[2] == expectedSharedObjects[2]);
    BOOST_TEST(sharedObjects[3] == expectedSharedObjects[3]);
    BOOST_TEST(sharedObjects[4] == expectedSharedObjects[4]);
    BOOST_TEST(sharedObjects[5] == expectedSharedObjects[5]);
    BOOST_TEST(sharedObjects[6] == expectedSharedObjects[6]);
    BOOST_TEST(sharedObjects[7] == expectedSharedObjects[7]);
    BOOST_TEST(sharedObjects[8] == expectedSharedObjects[8]);
}

void CreateDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

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
    BOOST_CHECK(exists(testDynamicBackendsSubDir5));
    BOOST_CHECK(exists(testDynamicBackendsSubDir6));
    BOOST_CHECK(exists(testDynamicBackendsSubDir7));
    BOOST_CHECK(!exists(testDynamicBackendsSubDir8));

    std::vector<std::string> backendPaths
    {
        testDynamicBackendsSubDir5,
        testDynamicBackendsSubDir6,
        testDynamicBackendsSubDir7,
        testDynamicBackendsSubDir8
    };
    std::vector<std::string> sharedObjects = DynamicBackendUtils::GetSharedObjects(backendPaths);
    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    BOOST_TEST(dynamicBackends.size() == 5);
    BOOST_TEST((dynamicBackends[0] != nullptr));
    BOOST_TEST((dynamicBackends[1] != nullptr));
    BOOST_TEST((dynamicBackends[2] != nullptr));
    BOOST_TEST((dynamicBackends[3] != nullptr));
    BOOST_TEST((dynamicBackends[4] != nullptr));

    // Duplicates are allowed here, they will be skipped later during the backend registration
    BOOST_TEST((dynamicBackends[0]->GetBackendId() == "TestValid2"));
    BOOST_TEST((dynamicBackends[1]->GetBackendId() == "TestValid3"));
    BOOST_TEST((dynamicBackends[2]->GetBackendId() == "TestValid2")); // From duplicate Arm_TestValid2_backend.so
    BOOST_TEST((dynamicBackends[3]->GetBackendId() == "TestValid2")); // From Arm_TestValid4_backend.so
    BOOST_TEST((dynamicBackends[4]->GetBackendId() == "TestValid5"));
}

void CreateDynamicBackendsNoPathsTestImpl()
{
    using namespace armnn;

    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends({});

    BOOST_TEST(dynamicBackends.empty());
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

    BOOST_TEST(dynamicBackends.empty());
}

void CreateDynamicBackendsMixedTypesTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    BOOST_CHECK(exists(testDynamicBackendsSubDir5));
    BOOST_CHECK(exists(testDynamicBackendsSubDir6));

    std::string testValidBackend2FilePath = GetTestFilePath(testDynamicBackendsSubDir5,
                                                            g_TestValidBackend2FileName);
    std::string testInvalidBackend8FilePath = GetTestFilePath(testDynamicBackendsSubDir5,
                                                              g_TestInvalidBackend8FileName);
    std::string testInvalidBackend9FilePath = GetTestFilePath(testDynamicBackendsSubDir6,
                                                              g_TestInvalidBackend9FileName);
    BOOST_CHECK(exists(testValidBackend2FilePath));
    BOOST_CHECK(exists(testInvalidBackend8FilePath));
    BOOST_CHECK(exists(testInvalidBackend9FilePath));

    std::vector<std::string> sharedObjects
    {
        testValidBackend2FilePath,   // Arm_TestValid2_backend.so     -> valid (basic backend name)
        testInvalidBackend8FilePath, // Arm_TestInvalid8_backend.so   -> not valid (invalid backend id)
        testInvalidBackend9FilePath, // Arm_TestInvalid9_backend.so   -> not valid (incompatible version)
        "InvalidSharedObject",       // The file does not exist
    };
    std::vector<DynamicBackendPtr> dynamicBackends = DynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    BOOST_TEST(dynamicBackends.size() == 1);
    BOOST_TEST((dynamicBackends[0] != nullptr));
    BOOST_TEST((dynamicBackends[0]->GetBackendId() == "TestValid2"));
}

void RegisterSingleDynamicBackendTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // Register one valid dynamic backend

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    BOOST_TEST(backendRegistry.Size() == 0);

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    BOOST_CHECK(exists(testDynamicBackendsSubDir5));

    std::string testValidBackend2FilePath = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend2FileName);
    BOOST_CHECK(exists(testValidBackend2FilePath));

    std::vector<std::string> sharedObjects{ testValidBackend2FilePath };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    BOOST_TEST(dynamicBackends.size() == 1);
    BOOST_TEST((dynamicBackends[0] != nullptr));

    BackendId dynamicBackendId = dynamicBackends[0]->GetBackendId();
    BOOST_TEST((dynamicBackendId == "TestValid2"));

    BackendVersion dynamicBackendVersion = dynamicBackends[0]->GetBackendVersion();
    BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));

    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    BOOST_TEST(backendRegistry.Size() == 1);
    BOOST_TEST(registeredBackendIds.size() == 1);

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    BOOST_TEST(backendIds.size() == 1);
    BOOST_TEST((backendIds.find(dynamicBackendId) != backendIds.end()));
    BOOST_TEST((registeredBackendIds.find(dynamicBackendId) != registeredBackendIds.end()));

    auto dynamicBackendFactoryFunction = backendRegistry.GetFactory(dynamicBackendId);
    BOOST_TEST((dynamicBackendFactoryFunction != nullptr));

    IBackendInternalUniquePtr dynamicBackend = dynamicBackendFactoryFunction();
    BOOST_TEST((dynamicBackend != nullptr));
    BOOST_TEST((dynamicBackend->GetId() == dynamicBackendId));
}

void RegisterMultipleDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // Register many valid dynamic backends

    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    BOOST_CHECK(exists(testDynamicBackendsSubDir5));
    BOOST_CHECK(exists(testDynamicBackendsSubDir6));

    std::string testValidBackend2FilePath = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend2FileName);
    std::string testValidBackend3FilePath = GetTestFilePath(testDynamicBackendsSubDir5, g_TestValidBackend3FileName);
    std::string testValidBackend5FilePath = GetTestFilePath(testDynamicBackendsSubDir6, g_TestValidBackend5FileName);
    BOOST_CHECK(exists(testValidBackend2FilePath));
    BOOST_CHECK(exists(testValidBackend3FilePath));
    BOOST_CHECK(exists(testValidBackend5FilePath));

    std::vector<std::string> sharedObjects
    {
        testValidBackend2FilePath,
        testValidBackend3FilePath,
        testValidBackend5FilePath
    };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    BOOST_TEST(dynamicBackends.size() == 3);
    BOOST_TEST((dynamicBackends[0] != nullptr));
    BOOST_TEST((dynamicBackends[1] != nullptr));
    BOOST_TEST((dynamicBackends[2] != nullptr));

    BackendId dynamicBackendId1 = dynamicBackends[0]->GetBackendId();
    BackendId dynamicBackendId2 = dynamicBackends[1]->GetBackendId();
    BackendId dynamicBackendId3 = dynamicBackends[2]->GetBackendId();
    BOOST_TEST((dynamicBackendId1 == "TestValid2"));
    BOOST_TEST((dynamicBackendId2 == "TestValid3"));
    BOOST_TEST((dynamicBackendId3 == "TestValid5"));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendVersion dynamicBackendVersion = dynamicBackends[i]->GetBackendVersion();
        BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));
    }

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    BOOST_TEST(backendRegistry.Size() == 0);

    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    BOOST_TEST(backendRegistry.Size() == 3);
    BOOST_TEST(registeredBackendIds.size() == 3);

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    BOOST_TEST(backendIds.size() == 3);
    BOOST_TEST((backendIds.find(dynamicBackendId1) != backendIds.end()));
    BOOST_TEST((backendIds.find(dynamicBackendId2) != backendIds.end()));
    BOOST_TEST((backendIds.find(dynamicBackendId3) != backendIds.end()));
    BOOST_TEST((registeredBackendIds.find(dynamicBackendId1) != registeredBackendIds.end()));
    BOOST_TEST((registeredBackendIds.find(dynamicBackendId2) != registeredBackendIds.end()));
    BOOST_TEST((registeredBackendIds.find(dynamicBackendId3) != registeredBackendIds.end()));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendId dynamicBackendId = dynamicBackends[i]->GetBackendId();

        auto dynamicBackendFactoryFunction = backendRegistry.GetFactory(dynamicBackendId);
        BOOST_TEST((dynamicBackendFactoryFunction != nullptr));

        IBackendInternalUniquePtr dynamicBackend = dynamicBackendFactoryFunction();
        BOOST_TEST((dynamicBackend != nullptr));
        BOOST_TEST((dynamicBackend->GetId() == dynamicBackendId));
    }
}

void RegisterMultipleInvalidDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

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
    BOOST_CHECK(exists(testDynamicBackendsSubDir9));

    std::string testInvalidBackend10FilePath = GetTestFilePath(testDynamicBackendsSubDir9,
                                                               g_TestInvalidBackend10FileName);
    std::string testInvalidBackend11FilePath = GetTestFilePath(testDynamicBackendsSubDir9,
                                                               g_TestInvalidBackend11FileName);
    BOOST_CHECK(exists(testInvalidBackend10FilePath));
    BOOST_CHECK(exists(testInvalidBackend11FilePath));

    std::vector<std::string> sharedObjects
    {
        testInvalidBackend10FilePath,
        testInvalidBackend11FilePath,
        "InvalidSharedObject"
    };
    std::vector<DynamicBackendPtr> dynamicBackends = TestDynamicBackendUtils::CreateDynamicBackends(sharedObjects);

    BOOST_TEST(dynamicBackends.size() == 2);
    BOOST_TEST((dynamicBackends[0] != nullptr));
    BOOST_TEST((dynamicBackends[1] != nullptr));

    BackendId dynamicBackendId1 = dynamicBackends[0]->GetBackendId();
    BackendId dynamicBackendId2 = dynamicBackends[1]->GetBackendId();
    BOOST_TEST((dynamicBackendId1 == ""));
    BOOST_TEST((dynamicBackendId2 == "Unknown"));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendVersion dynamicBackendVersion = dynamicBackends[i]->GetBackendVersion();
        BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));
    }

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    BOOST_TEST(backendRegistry.Size() == 0);

    // Check that no dynamic backend got registered
    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    BOOST_TEST(backendRegistry.Size() == 0);
    BOOST_TEST(registeredBackendIds.empty());
}

void RegisterMixedDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

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
    BOOST_CHECK(exists(testDynamicBackendsSubDir5));
    BOOST_CHECK(exists(testDynamicBackendsSubDir6));
    BOOST_CHECK(exists(testDynamicBackendsSubDir7));
    BOOST_CHECK(!exists(testDynamicBackendsSubDir8));
    BOOST_CHECK(exists(testDynamicBackendsSubDir9));

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
    BOOST_CHECK(exists(testValidBackend2FilePath));
    BOOST_CHECK(exists(testValidBackend3FilePath));
    BOOST_CHECK(exists(testValidBackend2DupFilePath));
    BOOST_CHECK(exists(testValidBackend4FilePath));
    BOOST_CHECK(exists(testValidBackend5FilePath));
    BOOST_CHECK(exists(testInvalidBackend8FilePath));
    BOOST_CHECK(exists(testInvalidBackend9FilePath));
    BOOST_CHECK(exists(testInvalidBackend10FilePath));
    BOOST_CHECK(exists(testInvalidBackend11FilePath));

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

    BOOST_TEST(dynamicBackends.size() == 7);
    BOOST_TEST((dynamicBackends[0] != nullptr));
    BOOST_TEST((dynamicBackends[1] != nullptr));
    BOOST_TEST((dynamicBackends[2] != nullptr));
    BOOST_TEST((dynamicBackends[3] != nullptr));
    BOOST_TEST((dynamicBackends[4] != nullptr));
    BOOST_TEST((dynamicBackends[5] != nullptr));
    BOOST_TEST((dynamicBackends[6] != nullptr));

    BackendId dynamicBackendId1 = dynamicBackends[0]->GetBackendId();
    BackendId dynamicBackendId2 = dynamicBackends[1]->GetBackendId();
    BackendId dynamicBackendId3 = dynamicBackends[2]->GetBackendId();
    BackendId dynamicBackendId4 = dynamicBackends[3]->GetBackendId();
    BackendId dynamicBackendId5 = dynamicBackends[4]->GetBackendId();
    BackendId dynamicBackendId6 = dynamicBackends[5]->GetBackendId();
    BackendId dynamicBackendId7 = dynamicBackends[6]->GetBackendId();
    BOOST_TEST((dynamicBackendId1 == "TestValid2"));
    BOOST_TEST((dynamicBackendId2 == "TestValid3"));
    BOOST_TEST((dynamicBackendId3 == "TestValid2")); // From duplicate Arm_TestValid2_backend.so
    BOOST_TEST((dynamicBackendId4 == "TestValid2")); // From Arm_TestValid4_backend.so
    BOOST_TEST((dynamicBackendId5 == "TestValid5"));
    BOOST_TEST((dynamicBackendId6 == ""));
    BOOST_TEST((dynamicBackendId7 == "Unknown"));

    for (size_t i = 0; i < dynamicBackends.size(); i++)
    {
        BackendVersion dynamicBackendVersion = dynamicBackends[i]->GetBackendVersion();
        BOOST_TEST(TestDynamicBackendUtils::IsBackendCompatible(dynamicBackendVersion));
    }

    // Dummy registry used for testing
    BackendRegistry backendRegistry;
    BOOST_TEST(backendRegistry.Size() == 0);

    std::vector<BackendId> expectedRegisteredbackendIds
    {
        "TestValid2",
        "TestValid3",
        "TestValid5"
    };

    BackendIdSet registeredBackendIds = TestDynamicBackendUtils::RegisterDynamicBackendsImplTest(backendRegistry,
                                                                                                 dynamicBackends);
    BOOST_TEST(backendRegistry.Size() == expectedRegisteredbackendIds.size());
    BOOST_TEST(registeredBackendIds.size() == expectedRegisteredbackendIds.size());

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    BOOST_TEST(backendIds.size() == expectedRegisteredbackendIds.size());
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        BOOST_TEST((backendIds.find(expectedRegisteredbackendId) != backendIds.end()));
        BOOST_TEST((registeredBackendIds.find(expectedRegisteredbackendId) != registeredBackendIds.end()));

        auto dynamicBackendFactoryFunction = backendRegistry.GetFactory(expectedRegisteredbackendId);
        BOOST_TEST((dynamicBackendFactoryFunction != nullptr));

        IBackendInternalUniquePtr dynamicBackend = dynamicBackendFactoryFunction();
        BOOST_TEST((dynamicBackend != nullptr));
        BOOST_TEST((dynamicBackend->GetId() == expectedRegisteredbackendId));
    }
}

void RuntimeEmptyTestImpl()
{
    using namespace armnn;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    BOOST_TEST(backendRegistry.Size() == 0);

    IRuntime::CreationOptions creationOptions;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const DeviceSpec& deviceSpec = *boost::polymorphic_downcast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    BOOST_TEST(supportedBackendIds.empty());

    BOOST_TEST(backendRegistry.Size() == 0);
}

void RuntimeDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains valid and invalid backends
    std::string testDynamicBackendsSubDir5 = GetTestSubDirectory(g_TestDynamicBackendsSubDir5);
    BOOST_CHECK(exists(testDynamicBackendsSubDir5));

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
    BOOST_TEST(backendRegistry.Size() == expectedRegisteredbackendIds.size());

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        BOOST_TEST((backendIds.find(expectedRegisteredbackendId) != backendIds.end()));
    }

    const DeviceSpec& deviceSpec = *boost::polymorphic_downcast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    BOOST_TEST(supportedBackendIds.size() == expectedRegisteredbackendIds.size());
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        BOOST_TEST((supportedBackendIds.find(expectedRegisteredbackendId) != supportedBackendIds.end()));
    }
}

void RuntimeDuplicateDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains valid, invalid and duplicate backends
    std::string testDynamicBackendsSubDir6 = GetTestSubDirectory(g_TestDynamicBackendsSubDir6);
    BOOST_CHECK(exists(testDynamicBackendsSubDir6));

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
    BOOST_TEST(backendRegistry.Size() == expectedRegisteredbackendIds.size());

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        BOOST_TEST((backendIds.find(expectedRegisteredbackendId) != backendIds.end()));
    }

    const DeviceSpec& deviceSpec = *boost::polymorphic_downcast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    BOOST_TEST(supportedBackendIds.size() == expectedRegisteredbackendIds.size());
    for (const BackendId& expectedRegisteredbackendId : expectedRegisteredbackendIds)
    {
        BOOST_TEST((supportedBackendIds.find(expectedRegisteredbackendId) != supportedBackendIds.end()));
    }
}

void RuntimeInvalidDynamicBackendsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains only invalid backends
    std::string testDynamicBackendsSubDir9 = GetTestSubDirectory(g_TestDynamicBackendsSubDir9);
    BOOST_CHECK(exists(testDynamicBackendsSubDir9));

    // Using the path override in CreationOptions to load some test dynamic backends
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = testDynamicBackendsSubDir9;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    BOOST_TEST(backendRegistry.Size() == 0);

    const DeviceSpec& deviceSpec = *boost::polymorphic_downcast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    BOOST_TEST(supportedBackendIds.empty());
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
    BOOST_TEST(backendRegistry.Size() == 0);

    const DeviceSpec& deviceSpec = *boost::polymorphic_downcast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    BOOST_TEST(supportedBackendIds.empty());
}

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

void CreateReferenceDynamicBackendTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // Swapping the backend registry storage for testing
    TestBackendRegistry testBackendRegistry;

    // This directory contains the reference dynamic backend
    std::string dynamicBackendsBaseDir = GetDynamicBackendsBasePath();
    std::string referenceDynamicBackendSubDir = GetTestSubDirectory(dynamicBackendsBaseDir,
                                                                    g_ReferenceDynamicBackendSubDir);
    BOOST_CHECK(exists(referenceDynamicBackendSubDir));

    // Check that the reference dynamic backend file exists
    std::string referenceBackendFilePath = GetTestFilePath(referenceDynamicBackendSubDir,
                                                           g_ReferenceBackendFileName);
    BOOST_CHECK(exists(referenceBackendFilePath));

    // Using the path override in CreationOptions to load the reference dynamic backend
    IRuntime::CreationOptions creationOptions;
    creationOptions.m_DynamicBackendsPath = referenceDynamicBackendSubDir;
    IRuntimePtr runtime = IRuntime::Create(creationOptions);

    const BackendRegistry& backendRegistry = BackendRegistryInstance();
    BOOST_TEST(backendRegistry.Size() == 1);

    BackendIdSet backendIds = backendRegistry.GetBackendIds();
    BOOST_TEST((backendIds.find("CpuRef") != backendIds.end()));

    const DeviceSpec& deviceSpec = *boost::polymorphic_downcast<const DeviceSpec*>(&runtime->GetDeviceSpec());
    BackendIdSet supportedBackendIds = deviceSpec.GetSupportedBackends();
    BOOST_TEST(supportedBackendIds.size() == 1);
    BOOST_TEST((supportedBackendIds.find("CpuRef") != supportedBackendIds.end()));

    // Get the factory function
    auto referenceDynamicBackendFactoryFunction = backendRegistry.GetFactory("CpuRef");
    BOOST_TEST((referenceDynamicBackendFactoryFunction != nullptr));

    // Use the factory function to create an instance of the reference backend
    IBackendInternalUniquePtr referenceDynamicBackend = referenceDynamicBackendFactoryFunction();
    BOOST_TEST((referenceDynamicBackend != nullptr));
    BOOST_TEST((referenceDynamicBackend->GetId() == "CpuRef"));

    // Test the backend instance by querying the layer support
    IBackendInternal::ILayerSupportSharedPtr referenceLayerSupport = referenceDynamicBackend->GetLayerSupport();
    BOOST_TEST((referenceLayerSupport != nullptr));

    TensorShape inputShape {  1, 16, 16, 16 };
    TensorShape outputShape{  1, 16, 16, 16 };
    TensorShape weightShape{ 16,  1,  1, 16 };
    TensorInfo inputInfo (inputShape,  DataType::Float32);
    TensorInfo outputInfo(outputShape, DataType::Float32);
    TensorInfo weightInfo(weightShape, DataType::Float32);
    Convolution2dDescriptor convolution2dDescriptor;
    bool referenceConvolution2dSupported =
            referenceLayerSupport->IsConvolution2dSupported(inputInfo,
                                                            outputInfo,
                                                            convolution2dDescriptor,
                                                            weightInfo,
                                                            EmptyOptional());
    BOOST_TEST(referenceConvolution2dSupported);

    // Test the backend instance by creating a workload
    IBackendInternal::IWorkloadFactoryPtr referenceWorkloadFactory = referenceDynamicBackend->CreateWorkloadFactory();
    BOOST_TEST((referenceWorkloadFactory != nullptr));

    // Create dummy settings for the workload
    Convolution2dQueueDescriptor convolution2dQueueDescriptor;
    WorkloadInfo workloadInfo
    {
        { inputInfo },
        { outputInfo }
    };
    convolution2dQueueDescriptor.m_Inputs.push_back(nullptr);
    auto weights = std::make_unique<ScopedCpuTensorHandle>(weightInfo);
    convolution2dQueueDescriptor.m_Weight = weights.get();

    // Create a convolution workload with the dummy settings
    auto workload = referenceWorkloadFactory->CreateConvolution2d(convolution2dQueueDescriptor, workloadInfo);
    BOOST_TEST((workload != nullptr));
    BOOST_TEST(workload.get() == boost::polymorphic_downcast<RefConvolution2dWorkload*>(workload.get()));
}

#endif
