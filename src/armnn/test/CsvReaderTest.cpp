//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "CsvReader.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
#include <boost/optional.hpp>

using namespace armnnUtils;

struct TestHelper {

    TestHelper()
    {
        BOOST_TEST_MESSAGE("setup fixture");
    }

    ~TestHelper()
    {
        BOOST_TEST_MESSAGE("teardown fixture");
        TearDown();
    }

    std::string CreateTempCsvFile()
    {
        boost::filesystem::path fileDir = boost::filesystem::temp_directory_path();
        boost::filesystem::path p{fileDir / boost::filesystem::unique_path("%%%%-%%%%-%%%%.csv")};
        try
        {
            boost::filesystem::ofstream ofs{p};
            ofs << "airplane, bicycle , bird , \"m,o,n,k,e,y\"\n";
            ofs << "banana, shoe, \"ice\"";
            ofs.close();
        } catch (std::exception &e)
        {
            std::cerr << "Unable to write to file at location [" << p.c_str() << "] : " << e.what() << std::endl;
            BOOST_TEST(false);
        }

        m_CsvFile = p;
        return p.string();
    }

    int CheckStringsMatch(CsvRow &row, unsigned int index, std::string expectedValue)
    {
        return row.values.at(index).compare(expectedValue);
    }

    void TearDown()
    {
        RemoveCsvFile();
    }

    void RemoveCsvFile()
    {
        if (m_CsvFile)
        {
            try
            {
                boost::filesystem::remove(*m_CsvFile);
            }
            catch (std::exception &e)
            {
                std::cerr << "Unable to delete file [" << *m_CsvFile << "] : " << e.what() << std::endl;
                BOOST_TEST(false);
            }
        }
    }

    boost::optional<boost::filesystem::path> m_CsvFile;
};

BOOST_AUTO_TEST_SUITE(CsvReaderTest)

BOOST_FIXTURE_TEST_CASE(TestParseVector, TestHelper)
{
    CsvReader reader;
    std::vector<std::string> csvStrings;
    csvStrings.reserve(2);
    csvStrings.push_back("airplane, automobile , bird , \"c,a,t\"");
    csvStrings.push_back("banana, shoe, \"ice\"");

    std::vector<CsvRow> row = reader.ParseVector(csvStrings);
    CsvRow row1 = row[0];
    CsvRow row2 = row[1];

    BOOST_CHECK(row.size() == 2);

    BOOST_CHECK(row1.values.size() == 4);
    BOOST_CHECK(CheckStringsMatch(row1, 0, "airplane") == 0);
    BOOST_CHECK(CheckStringsMatch(row1, 1, "automobile") == 0);
    BOOST_CHECK(CheckStringsMatch(row1, 2, "bird") == 0);
    BOOST_CHECK(CheckStringsMatch(row1, 3, "c,a,t") == 0);

    BOOST_CHECK(row2.values.size() == 3);
    BOOST_CHECK(CheckStringsMatch(row2, 0, "banana") == 0);
    BOOST_CHECK(CheckStringsMatch(row2, 1, "shoe") == 0);
    BOOST_CHECK(CheckStringsMatch(row2, 2, "ice") == 0);
}

BOOST_FIXTURE_TEST_CASE(TestLoadingFileFromDisk, TestHelper)
{
    CsvReader reader;
    std::string theFilePath = TestHelper::CreateTempCsvFile();

    std::vector<CsvRow> row = reader.ParseFile(theFilePath);
    CsvRow row1 = row[0];
    CsvRow row2 = row[1];

    BOOST_CHECK(row.size() == 2);

    BOOST_CHECK(row1.values.size() == 4);
    BOOST_CHECK(CheckStringsMatch(row1, 0, "airplane") == 0);
    BOOST_CHECK(CheckStringsMatch(row1, 1, "bicycle") == 0);
    BOOST_CHECK(CheckStringsMatch(row1, 2, "bird") == 0);
    BOOST_CHECK(CheckStringsMatch(row1, 3, "m,o,n,k,e,y") == 0);

    BOOST_CHECK(row2.values.size() == 3);
    BOOST_CHECK(CheckStringsMatch(row2, 0, "banana") == 0);
    BOOST_CHECK(CheckStringsMatch(row2, 1, "shoe") == 0);
    BOOST_CHECK(CheckStringsMatch(row2, 2, "ice") == 0);
}

BOOST_AUTO_TEST_SUITE_END()
