//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../InferenceTest.hpp"
#include "../MnistDatabase.hpp"
#include "armnnOnnxParser/IOnnxParser.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape({ 1, 1, 28, 28 });

    int retVal = EXIT_FAILURE;
    try
    {
        using DataType = float;
        using DatabaseType = MnistDatabase;
        using ParserType = armnnOnnxParser::IOnnxParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                     argc, argv, "mnist_onnx.onnx", true,
                     "Input3", "Plus214_Output_0", { 0, 1, 2, 3, 4},
                     [](const char* dataDir, const ModelType&) {
                         return DatabaseType(dataDir, true);
                     },
                     &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: OnnxMnist-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
