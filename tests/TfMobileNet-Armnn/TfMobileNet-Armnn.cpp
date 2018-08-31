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
            {"Dog.jpg", 209},
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 209:Labrador retriever 0.949995
            // 160:Rhodesian ridgeback 0.0270182
            // 208:golden retriever 0.0192866
            // 853:tennis ball 0.000470382
            // 239:Greater Swiss Mountain dog 0.000464451
            {"Cat.jpg", 283},
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 283:tiger cat 0.579016
            // 286:Egyptian cat 0.319676
            // 282:tabby, tabby cat 0.0873346
            // 288:lynx, catamount 0.011163
            // 289:leopard, Panthera pardus 0.000856755
            {"shark.jpg", 3},
            // Top five predictions in tensorflow:
            // -----------------------------------
            // 3:great white shark, white shark, ... 0.996926
            // 4:tiger shark, Galeocerdo cuvieri 0.00270528
            // 149:killer whale, killer, orca, ... 0.000121848
            // 395:sturgeon 7.78977e-05
            // 5:hammerhead, hammerhead shark 6.44127e-055
        };

        armnn::TensorShape inputTensorShape({ 1, 224, 224, 3  });

        using DataType = float;
        using DatabaseType = ImagePreprocessor<float>;
        using ParserType = armnnTfParser::ITfParser;
        using ModelType = InferenceModel<ParserType, DataType>;

        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<DatabaseType, ParserType>(
                     argc, argv,
                     "mobilenet_v1_1.0_224_fp32.pb", // model name
                     true,                           // model is binary
                     "input", "output",              // input and output tensor names
                     { 0, 1, 2 },                    // test images to test with as above
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
