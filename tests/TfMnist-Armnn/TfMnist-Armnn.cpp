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
    return armnn::test::ClassifierInferenceTestMain<MnistDatabase, armnnTfParser::ITfParser>(
        argc, argv, "simple_mnist_tf.prototxt", false,
        "Placeholder", "Softmax", { 0, 1, 2, 3, 4 },
        [](const char* dataDir) { return MnistDatabase(dataDir, true); },
        &inputTensorShape);
}
