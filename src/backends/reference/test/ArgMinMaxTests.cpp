//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/workloads/ArgMinMax.hpp>

#include <doctest/doctest.h>

TEST_SUITE("RefArgMinMax")
{
TEST_CASE("ArgMinTest")
{
    const armnn::TensorInfo inputInfo({ 1, 2, 3 } , armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3 }, armnn::DataType::Signed64);

    std::vector<float> inputValues({ 1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f});
    std::vector<int64_t> outputValues(outputInfo.GetNumElements());
    std::vector<int64_t> expectedValues({ 0, 1, 0 });

    ArgMinMax(*armnn::MakeDecoder<float>(inputInfo, inputValues.data()),
               outputValues.data(),
               inputInfo,
               outputInfo,
               armnn::ArgMinMaxFunction::Min,
               -2);

    CHECK(std::equal(outputValues.begin(), outputValues.end(), expectedValues.begin(), expectedValues.end()));

}

TEST_CASE("ArgMaxTest")
{
    const armnn::TensorInfo inputInfo({ 1, 2, 3 } , armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo({ 1, 3 }, armnn::DataType::Signed64);

    std::vector<float> inputValues({ 1.0f, 5.0f, 3.0f, 4.0f, 2.0f, 6.0f });
    std::vector<int64_t> outputValues(outputInfo.GetNumElements());
    std::vector<int64_t> expectedValues({ 1, 0, 1 });

    ArgMinMax(*armnn::MakeDecoder<float>(inputInfo, inputValues.data()),
               outputValues.data(),
               inputInfo,
               outputInfo,
               armnn::ArgMinMaxFunction::Max,
               -2);

    CHECK(std::equal(outputValues.begin(), outputValues.end(), expectedValues.begin(), expectedValues.end()));

}

}