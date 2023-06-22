//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <ExecuteNetwork/FileComparisonExecutor.hpp>
#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
namespace
{

namespace fs = std::filesystem;

TEST_SUITE("FileComparisonExecutorTests")
{

    TEST_CASE("EmptyComparisonThrowsException")
    {
        ExecuteNetworkParams params;
        FileComparisonExecutor classToTest(params);
        // The comparison file is not set in the parameters. This should throw an exception.
        CHECK_THROWS_AS(classToTest.Execute(), armnn::InvalidArgumentException);
    }

    TEST_CASE("InvalidComparisonFilesThrowsException")
    {
        ExecuteNetworkParams params;
        params.m_ComparisonFile = "Balh,Blah,Blah";
        FileComparisonExecutor classToTest(params);
        // None of the files in the parameter exist.
        CHECK_THROWS_AS(classToTest.Execute(), armnn::FileNotFoundException);
    }

    TEST_CASE("ComparisonFileIsEmpty")
    {
        std::filesystem::path fileName = fs::temp_directory_path().append("ComparisonFileIsEmpty.tmp");
        std::fstream tmpFile;
        tmpFile.open(fileName, std::ios::out);
        ExecuteNetworkParams params;
        params.m_ComparisonFile = fileName;
        FileComparisonExecutor classToTest(params);
        // The comparison file is empty. This exception should happen in ExtractHeader when it realises it
        // can't read a header.
        CHECK_THROWS_AS(classToTest.Execute(), armnn::ParseException);
        tmpFile.close();
        std::filesystem::remove(fileName);
    }

    TEST_CASE("ComparisonFileHasValidHeaderAndData")
    {
        std::filesystem::path fileName = fs::temp_directory_path().append("ComparisonFileHasValidHeaderAndData.tmp");
        std::fstream tmpFile;
        tmpFile.open(fileName, std::ios::out);
        // Write a valid header.
        tmpFile << "TensorName, Float32 : 1.1000";
        tmpFile.close();
        ExecuteNetworkParams params;
        params.m_ComparisonFile = fileName;
        FileComparisonExecutor classToTest(params);
        // The read in tensor should consist of 1 float.
        std::vector<const void*> results = classToTest.Execute();
        std::filesystem::remove(fileName);
        // Should be one tensor in the data.
        CHECK_EQ(1, results.size());
        // We expect there to be 1 element of value 1.1f.
        const float* floatPtr = static_cast<const float*>(results[0]);
        CHECK_EQ(*floatPtr, 1.1f);
    }


}    // End of TEST_SUITE("FileComparisonExecutorTests")

}    // anonymous namespace