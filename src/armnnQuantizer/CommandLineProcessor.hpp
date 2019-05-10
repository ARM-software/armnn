//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <string>
#include <iostream>
#include <vector>
#include "QuantizationDataSet.hpp"

namespace armnnQuantizer
{

// parses the command line to extract
// * the input file -f containing the serialized fp32 ArmNN input graph (must exist...and be a input graph file)
// * the csv file -c <optional> detailing the paths for RAW input tensors to use for refinement
// * the directory -d to place the output file into (must already exist and be writable)
// * the name of the file -o the quantized ArmNN input graph will be written to (must not already exist)
// * LATER: the min and max overrides to be applied to the inputs
//          specified as -i <int> (input id) -n <float> (minimum) -x <float> (maximum)
//          multiple sets of -i, -n, -x can appear on the command line but they must match
//          in number i.e. a -n and -x for each -i and the id of the input must correspond
//          to an input layer in the fp32 graph when it is loaded.
class CommandLineProcessor
{
public:
    bool ProcessCommandLine(int argc, char* argv[]);

    std::string GetInputFileName() {return m_InputFileName;}
    std::string GetCsvFileName() {return m_CsvFileName;}
    std::string GetCsvFileDirectory() {return m_CsvFileDirectory;}
    std::string GetOutputDirectoryName() {return m_OutputDirectory;}
    std::string GetOutputFileName() {return m_OutputFileName;}
    std::string GetQuantizationScheme() {return m_QuantizationScheme;}
    QuantizationDataSet GetQuantizationDataSet() {return m_QuantizationDataSet;}
    bool HasPreservedDataType() {return m_PreserveDataType;}
    bool HasQuantizationData() {return !m_QuantizationDataSet.IsEmpty();}

protected:
    std::string m_InputFileName;
    std::string m_CsvFileName;
    std::string m_CsvFileDirectory;
    std::string m_OutputDirectory;
    std::string m_OutputFileName;
    std::string m_QuantizationScheme;
    QuantizationDataSet m_QuantizationDataSet;
    bool m_PreserveDataType;
};

} // namespace armnnQuantizer

