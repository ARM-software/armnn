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
            {"Dog.jpg", 209},
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 209:Labrador retriever 0.46392533
            // 160:Rhodesian ridgeback 0.29911423
            // 208:golden retriever 0.108059585
            // 169:redbone 0.033753652
            // 274:dingo, warrigal, warragal, ... 0.01232666

            {"Cat.jpg", 283},
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 283:tiger cat 0.6508582
            // 286:Egyptian cat 0.2604343
            // 282:tabby, tabby cat 0.028786005
            // 288:lynx, catamount 0.020673484
            // 40:common iguana, iguana, ... 0.0080499435

            {"shark.jpg", 3},
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 3:great white shark, white shark, ... 0.96672016
            // 4:tiger shark, Galeocerdo cuvieri 0.028302953
            // 149:killer whale, killer, orca, ... 0.0020228163
            // 5:hammerhead, hammerhead shark 0.0017547971
            // 150:dugong, Dugong dugon 0.0003968083
        };

        armnn::TensorShape inputTensorShape({ 1, 224, 224, 3  });

        using DataType = float;
        using DatabaseType = ImagePreprocessor<float>;
        using ParserType = armnnTfParser::ITfParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                     argc, argv,
                     "mobilenet_v1_1.0_224_frozen.pb",              // model name
                     true,                                          // model is binary
                     "input", "MobilenetV1/Predictions/Reshape_1",  // input and output tensor names
                     { 0, 1, 2 },                                   // test images to test with as above
                     [&imageSet](const char* dataDir, const ModelType&) {
                         // This creates a 224x224x3 NHWC float tensor to pass to Armnn
                         return DatabaseType(
                             dataDir,
                             224,
                             224,
                             imageSet);
                     },
                     &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: TfMobileNet-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
