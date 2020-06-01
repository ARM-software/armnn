//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 208:golden retriever 0.57466376
            // 209:Labrador retriever 0.30202731
            // 853:tennis ball 0.0060001756
            // 223:kuvasz 0.0053707925
            // 160:Rhodesian ridgeback 0.0018179063

            { "Cat.jpg", 283 },
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 283:tiger cat 0.4667799
            // 282:tabby, tabby cat 0.32511184
            // 286:Egyptian cat 0.1038616
            // 288:lynx, catamount 0.0017019814
            // 284:Persian cat 0.0011340436

            { "shark.jpg", 3 },
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 3:great white shark, white shark, ... 0.98808634
            // 148:grey whale, gray whale, ... 0.00070245547
            // 234:Bouvier des Flandres, ... 0.00024639888
            // 149:killer whale, killer, ... 0.00014115588
            // 95:hummingbird 0.00011129203
        };

        armnn::TensorShape inputTensorShape({ 1, 299, 299, 3 });

        using DataType = float;
        using DatabaseType = ImagePreprocessor<float>;
        using ParserType = armnnTfParser::ITfParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: InferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                    argc, argv, "inception_v3_2016_08_28_frozen.pb", true,
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
