//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "../InferenceTest.hpp"
#include "../MnistDatabase.hpp"
#include "armnnTfParser/ITfParser.hpp"

int main(int argc, char* argv[])
{
    armnn::TensorShape inputTensorShape({ 1, 784, 1, 1 });

    int retVal = EXIT_FAILURE;
    try
    {
        // Coverity fix: ClassifierInferenceTestMain() may throw uncaught exceptions.
        retVal = armnn::test::ClassifierInferenceTestMain<MnistDatabase, armnnTfParser::ITfParser>(
                     argc, argv, "simple_mnist_tf.prototxt", false,
                     "Placeholder", "Softmax", { 0, 1, 2, 3, 4 },
                     [](const char* dataDir) { return MnistDatabase(dataDir, true); },
                     &inputTensorShape);
    }
    catch (const std::exception& e)
    {
        // Coverity fix: BOOST_LOG_TRIVIAL (typically used to report errors) may throw an
        // exception of type std::length_error.
        // Using stderr instead in this context as there is no point in nesting try-catch blocks here.
        std::cerr << "WARNING: TfMnist-Armnn: An error has occurred when running "
                     "the classifier inference tests: " << e.what() << std::endl;
    }
    return retVal;
}
