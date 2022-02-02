//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnnTestUtils/TensorCopyUtils.hpp>
#include <armnnTestUtils/WorkloadTestUtils.hpp>

#include <armnn/utility/NumericCast.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

struct ActivationFixture
{
    ActivationFixture()
    {
        output.resize(batchSize * channels * height * width);
        outputExpected.resize(batchSize * channels * height * width);
        input.resize(batchSize * channels * height * width);

        unsigned int inputShape[]  = { batchSize, channels, height, width };
        unsigned int outputShape[] = { batchSize, channels, height, width };

        inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
        outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

        input = MakeRandomTensor<float>(inputTensorInfo, 21453);
    }

    unsigned int width     = 17;
    unsigned int height    = 29;
    unsigned int channels  = 2;
    unsigned int batchSize = 5;

    std::vector<float> output;
    std::vector<float> outputExpected;
    std::vector<float> input;

    armnn::TensorInfo inputTensorInfo;
    armnn::TensorInfo outputTensorInfo;

    // Parameters used by some of the activation functions.
    float a = 0.234f;
    float b = -12.345f;
};


struct PositiveActivationFixture : public ActivationFixture
{
    PositiveActivationFixture()
    {
        input = MakeRandomTensor<float>(inputTensorInfo, 2342423, 0.0f, 1.0f);
    }
};