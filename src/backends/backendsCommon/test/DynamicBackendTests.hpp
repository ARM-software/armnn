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

static std::string g_TestDynamicBackendsFileParsingSubDir1  = "backendsTestPath1/";
static std::string g_TestDynamicBackendsFileParsingSubDir2  = "backendsTestPath2/";
static std::string g_TestDynamicBackendsFileParsingSubDir3  = "backendsTestPath3/";
static std::string g_TestDynamicBackendsFileParsingSubDir4  = "backendsTestPath4/";

std::string GetTestDirectoryBasePath()
{
    using namespace boost::filesystem;

    path programLocation = boost::dll::program_location().parent_path();
    path sharedObjectPath = programLocation.append(g_TestBaseDir);
    BOOST_CHECK(exists(sharedObjectPath));

    return sharedObjectPath.string();
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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestValidTestDynamicBackendFileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend1FileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend2FileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend3FileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend4FileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend5FileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend6FileName);

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

    std::string testSubDirectory = GetTestSubDirectory(g_TestDynamicBackendSubDir);
    std::string sharedObjectFilePath = GetTestFilePath(testSubDirectory, g_TestInvalidTestDynamicBackend7FileName);

    void* sharedObjectHandle = nullptr;
    BOOST_CHECK_NO_THROW(sharedObjectHandle = DynamicBackendUtils::OpenHandle(sharedObjectFilePath));
    BOOST_TEST((sharedObjectHandle != nullptr));

    std::unique_ptr<DynamicBackend> dynamicBackend;
    BOOST_CHECK_THROW(dynamicBackend.reset(new DynamicBackend(sharedObjectHandle)), RuntimeException);
    BOOST_TEST((dynamicBackend == nullptr));
}

void GetBackendPathsTestImpl()
{
    using namespace armnn;
    using namespace boost::filesystem;

    // The test covers four directories:
    // <unit test path>/src/backends/backendsCommon/test/
    //                                                ├─ backendsTestPath1/   -> existing, contains files
    //                                                ├─ backendsTestPath2/   -> existing, contains files
    //                                                ├─ backendsTestPath3/   -> existing, but empty
    //                                                └─ backendsTestPath4/   -> not existing

    std::string subDir1 = GetTestSubDirectory(g_TestDynamicBackendsFileParsingSubDir1);
    std::string subDir2 = GetTestSubDirectory(g_TestDynamicBackendsFileParsingSubDir2);
    std::string subDir3 = GetTestSubDirectory(g_TestDynamicBackendsFileParsingSubDir3);
    std::string subDir4 = GetTestSubDirectory(g_TestDynamicBackendsFileParsingSubDir4);

    BOOST_CHECK(exists(subDir1));
    BOOST_CHECK(exists(subDir2));
    BOOST_CHECK(exists(subDir3));
    BOOST_CHECK(!exists(subDir4));

    class TestDynamicBackendUtils : public DynamicBackendUtils
    {
    public:
        static std::vector<std::string> GetBackendPathsImplTest(const std::string& path)
        {
            return GetBackendPathsImpl(path);
        }
    };

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

    std::string subDir1 = GetTestSubDirectory(g_TestDynamicBackendsFileParsingSubDir1);
    std::string subDir4 = GetTestSubDirectory(g_TestDynamicBackendsFileParsingSubDir4);

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
