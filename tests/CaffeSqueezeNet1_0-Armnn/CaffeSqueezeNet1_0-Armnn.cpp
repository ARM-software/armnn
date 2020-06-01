//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../InferenceTest.hpp"
#include "../CaffePreprocessor.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    using DataType = float;
    using DatabaseType = CaffePreprocessor;
    using ParserType = armnnCaffeParser::ICaffeParser;
    using ModelType = InferenceModel<ParserType, DataType>;

    return armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
        argc, argv, "squeezenet.caffemodel", true,
        "input", "prob", { 0 },
        [](const char* dataDir, const ModelType &) { return CaffePreprocessor(dataDir); });
}
