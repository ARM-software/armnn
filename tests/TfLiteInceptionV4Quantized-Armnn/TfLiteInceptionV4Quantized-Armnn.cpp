//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "../InferenceTest.hpp"
#include "../ImagePreprocessor.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"

using namespace armnnTfLiteParser;

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        std::vector<ImageSet> imageSet =
        {
            {"Dog.jpg", 209},
            {"Cat.jpg", 283},
            {"shark.jpg", 3},

        };

        armnn::TensorShape inputTensorShape({ 1, 299, 299, 3  });

        using DataType = uint8_t;
        using DatabaseType = ImagePreprocessor<DataType>;
        using ParserType = armnnTfLiteParser::ITfLiteParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType,
                                                          ParserType>(
                     argc, argv,
                     "inception_v4_299_quant.tflite",     // model name
                     true,                                // model is binary
                     "input",                             // input tensor name
                     "InceptionV4/Logits/Predictions",    // output tensor name
                     { 0, 1, 2 },                         // test images to test with as above
                     [&imageSet](const char* dataDir, const ModelType &) {
                         // we need to get the input quantization parameters from
                         // the parsed model
                         return DatabaseType(
                             dataDir,
                             299,
                             299,
                             imageSet,
                             1);
                     },
                     &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: " << *argv << ": An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
