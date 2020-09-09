//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "TensorCopyUtils.hpp"
#include "WorkloadTestUtils.hpp"

#include <armnn/utility/NumericCast.hpp>

#include <test/TensorHelpers.hpp>

#include <boost/multi_array.hpp>

struct ActivationFixture
{
    ActivationFixture()
    {
        auto boostArrayExtents = boost::extents
            [armnn::numeric_cast<boost::multi_array_types::extent_gen::index>(batchSize)]
            [armnn::numeric_cast<boost::multi_array_types::extent_gen::index>(channels)]
            [armnn::numeric_cast<boost::multi_array_types::extent_gen::index>(height)]
            [armnn::numeric_cast<boost::multi_array_types::extent_gen::index>(width)];
        output.resize(boostArrayExtents);
        outputExpected.resize(boostArrayExtents);
        input.resize(boostArrayExtents);

        unsigned int inputShape[]  = { batchSize, channels, height, width };
        unsigned int outputShape[] = { batchSize, channels, height, width };

        inputTensorInfo = armnn::TensorInfo(4, inputShape, armnn::DataType::Float32);
        outputTensorInfo = armnn::TensorInfo(4, outputShape, armnn::DataType::Float32);

        input = MakeRandomTensor<float, 4>(inputTensorInfo, 21453);
    }

    unsigned int width     = 17;
    unsigned int height    = 29;
    unsigned int channels  = 2;
    unsigned int batchSize = 5;

    boost::multi_array<float, 4> output;
    boost::multi_array<float, 4> outputExpected;
    boost::multi_array<float, 4> input;

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
        input = MakeRandomTensor<float, 4>(inputTensorInfo, 2342423, 0.0f, 1.0f);
    }
};