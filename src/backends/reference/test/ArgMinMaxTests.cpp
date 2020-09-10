//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/workloads/ArgMinMax.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(RefArgMinMax)

BOOST_AUTO_TEST_CASE(ArgMinTest)
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

    BOOST_CHECK_EQUAL_COLLECTIONS(outputValues.begin(),
                                  outputValues.end(),
                                  expectedValues.begin(),
                                  expectedValues.end());

}

BOOST_AUTO_TEST_CASE(ArgMaxTest)
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

    BOOST_CHECK_EQUAL_COLLECTIONS(outputValues.begin(),
                                  outputValues.end(),
                                  expectedValues.begin(),
                                  expectedValues.end());

}

BOOST_AUTO_TEST_SUITE_END()