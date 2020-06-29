//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#include "../QuantizationDataSet.hpp"

#include <armnn/Optional.hpp>
#include <Filesystem.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>


using namespace armnnQuantizer;

struct CsvTestHelper {

    CsvTestHelper()
    {
        BOOST_TEST_MESSAGE("setup fixture");
    }

    ~CsvTestHelper()
    {
        BOOST_TEST_MESSAGE("teardown fixture");
        TearDown();
    }

    std::string CreateTempCsvFile(std::map<int, std::vector<float>> csvData)
    {
        fs::path fileDir = fs::temp_directory_path();
        fs::path p = armnnUtils::Filesystem::NamedTempFile("Armnn-QuantizationCreateTempCsvFileTest-TempFile.csv");

        fs::path tensorInput1{fileDir / "input_0_0.raw"};
        fs::path tensorInput2{fileDir / "input_1_0.raw"};
        fs::path tensorInput3{fileDir / "input_2_0.raw"};

        try
        {
            std::ofstream ofs{p};

            std::ofstream ofs1{tensorInput1};
            std::ofstream ofs2{tensorInput2};
            std::ofstream ofs3{tensorInput3};


            for(auto entry : csvData.at(0))
            {
                ofs1 << entry << " ";
            }
            for(auto entry : csvData.at(1))
            {
                ofs2 << entry << " ";
            }
            for(auto entry : csvData.at(2))
            {
                ofs3 << entry << " ";
            }

            ofs << "0, 0, " << tensorInput1.c_str() << std::endl;
            ofs << "2, 0, " << tensorInput3.c_str() << std::endl;
            ofs << "1, 0, " << tensorInput2.c_str() << std::endl;

            ofs.close();
            ofs1.close();
            ofs2.close();
            ofs3.close();
        }
        catch (std::exception &e)
        {
            std::cerr << "Unable to write to file at location [" << p.c_str() << "] : " << e.what() << std::endl;
            BOOST_TEST(false);
        }

        m_CsvFile = p;
        return p.string();
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
                fs::remove(m_CsvFile.value());
            }
            catch (std::exception &e)
            {
                std::cerr << "Unable to delete file [" << m_CsvFile.value() << "] : " << e.what() << std::endl;
                BOOST_TEST(false);
            }
        }
    }

    armnn::Optional<fs::path> m_CsvFile;
};


BOOST_AUTO_TEST_SUITE(QuantizationDataSetTests)

BOOST_FIXTURE_TEST_CASE(CheckDataSet, CsvTestHelper)
{

    std::map<int, std::vector<float>> csvData;
    csvData.insert(std::pair<int, std::vector<float>>(0, { 0.111111f, 0.222222f, 0.333333f }));
    csvData.insert(std::pair<int, std::vector<float>>(1, { 0.444444f, 0.555555f, 0.666666f }));
    csvData.insert(std::pair<int, std::vector<float>>(2, { 0.777777f, 0.888888f, 0.999999f }));

    std::string myCsvFile = CsvTestHelper::CreateTempCsvFile(csvData);
    QuantizationDataSet dataSet(myCsvFile);
    BOOST_TEST(!dataSet.IsEmpty());

    int csvRow = 0;
    for(armnnQuantizer::QuantizationInput input : dataSet)
    {
        BOOST_TEST(input.GetPassId() == csvRow);

        BOOST_TEST(input.GetLayerBindingIds().size() == 1);
        BOOST_TEST(input.GetLayerBindingIds()[0] == 0);
        BOOST_TEST(input.GetDataForEntry(0).size() == 3);

        // Check that QuantizationInput data for binding ID 0 corresponds to float values
        // used for populating the CSV file using by QuantizationDataSet
        BOOST_TEST(input.GetDataForEntry(0).at(0) == csvData.at(csvRow).at(0));
        BOOST_TEST(input.GetDataForEntry(0).at(1) == csvData.at(csvRow).at(1));
        BOOST_TEST(input.GetDataForEntry(0).at(2) == csvData.at(csvRow).at(2));
        ++csvRow;
    }
}

BOOST_AUTO_TEST_SUITE_END();