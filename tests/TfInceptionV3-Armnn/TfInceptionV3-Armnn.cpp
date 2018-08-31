//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../ImagePreprocessor.hpp"
#include "armnnTfParser/ITfParser.hpp"

int main(int argc, char* argv[])
{
    int retVal = EXIT_FAILURE;
    try
    {
        // Coverity fix: The following code may throw an exception of type std::length_error.
        std::vector<ImageSet> imageSet =
        {
            { "Dog.jpg", 208 },
            { "Cat.jpg", 283 },
            { "shark.jpg", 3 },
        };

        armnn::TensorShape inputTensorShape({ 1, 299, 299, 3 });

        using DataType = float;
        using DatabaseType = ImagePreprocessor<float>;
        using ParserType = armnnTfParser::ITfParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: InferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                    argc, argv, "inception_v3_2016_08_28_frozen_transformed.pb", true,
                    "input", "InceptionV3/Predictions/Reshape_1", { 0, 1, 2, },
                    [&imageSet](const char* dataDir, const ModelType&) {
                        return DatabaseType(dataDir, 299, 299, imageSet);
                    },
                    &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: TfInceptionV3-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
