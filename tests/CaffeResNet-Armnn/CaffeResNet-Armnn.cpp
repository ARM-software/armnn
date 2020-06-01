//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../InferenceTest.hpp"
#include "../CaffePreprocessor.hpp"
#include "armnnCaffeParser/ICaffeParser.hpp"

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        // Coverity fix: The following code may throw an exception of type std::length_error.
        std::vector<ImageSet> imageSet =
        {
            {"ILSVRC2012_val_00000018.JPEG",  21 },
            {"shark.jpg", 2}
        };

        armnn::TensorShape inputTensorShape({ 1, 3, 224, 224 });

        using DataType = float;
        using DatabaseType = CaffePreprocessor;
        using ParserType = armnnCaffeParser::ICaffeParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                    argc, argv, "ResNet_50_ilsvrc15_model.caffemodel", true,
                    "data", "prob", { 0, 1 },
                    [&imageSet](const char* dataDir, const ModelType&) {
                        return DatabaseType(dataDir, 224, 224, imageSet);
                    }, &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: CaffeResNet-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
