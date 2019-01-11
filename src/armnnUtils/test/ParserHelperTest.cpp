//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../ParserHelper.hpp"

#include <boost/test/unit_test.hpp>

#include "armnn/Types.hpp"

using namespace armnn;
using namespace armnnUtils;

BOOST_AUTO_TEST_SUITE(ParserHelperSuite)

BOOST_AUTO_TEST_CASE(CalculateReducedOutputTensoInfoTest)
{
    bool keepDims = false;

    unsigned int inputShape[] = { 2, 3, 4 };
    TensorInfo inputTensorInfo(3, &inputShape[0], DataType::Float32);

    // Reducing all dimensions results in one single output value (one dimension)
    unsigned int axisShape1[] = { 3 };
    std::set<unsigned int> axisData1 = { 0, 1, 2 };
    TensorInfo axisTensorInfo1(1, &axisShape1[0], DataType::Signed32);

    TensorInfo outputTensorInfo1;

    CalculateReducedOutputTensoInfo(inputTensorInfo, axisTensorInfo1, axisData1,
    keepDims, outputTensorInfo1);

    BOOST_ASSERT(outputTensorInfo1.GetNumDimensions() == 1);
    BOOST_ASSERT(outputTensorInfo1.GetShape()[0] == 1);

    // Reducing dimension 0 results in a 3x4 size tensor (one dimension)
    unsigned int axisShape2[] = { 1 };
    std::set<unsigned int> axisData2 = { 0 };
    TensorInfo axisTensorInfo2(1, &axisShape2[0], DataType::Signed32);

    TensorInfo outputTensorInfo2;

    CalculateReducedOutputTensoInfo(inputTensorInfo, axisTensorInfo2, axisData2,
    keepDims, outputTensorInfo2);

    BOOST_ASSERT(outputTensorInfo2.GetNumDimensions() == 1);
    BOOST_ASSERT(outputTensorInfo2.GetShape()[0] == 12);

    // Reducing dimensions 0,1 results in a 4 size tensor (one dimension)
    unsigned int axisShape3[] = { 2 };
    std::set<unsigned int> axisData3 = { 0, 1 };
    TensorInfo axisTensorInfo3(1, &axisShape3[0], DataType::Signed32);

    TensorInfo outputTensorInfo3;

    CalculateReducedOutputTensoInfo(inputTensorInfo, axisTensorInfo3, axisData3,
    keepDims, outputTensorInfo3);

    BOOST_ASSERT(outputTensorInfo3.GetNumDimensions() == 1);
    BOOST_ASSERT(outputTensorInfo3.GetShape()[0] == 4);

    // Reducing dimension 0 results in a { 1, 3, 4 } dimension tensor
    keepDims = true;
    unsigned int axisShape4[] = { 1 };
    std::set<unsigned int> axisData4 = { 0 };
    TensorInfo axisTensorInfo4(1, &axisShape4[0], DataType::Signed32);

    TensorInfo outputTensorInfo4;

    CalculateReducedOutputTensoInfo(inputTensorInfo, axisTensorInfo4, axisData4,
    keepDims, outputTensorInfo4);

    BOOST_ASSERT(outputTensorInfo4.GetNumDimensions() == 3);
    BOOST_ASSERT(outputTensorInfo4.GetShape()[0] == 1);
    BOOST_ASSERT(outputTensorInfo4.GetShape()[1] == 3);
    BOOST_ASSERT(outputTensorInfo4.GetShape()[2] == 4);

    // Reducing dimension 1, 2 results in a { 2, 1, 1 } dimension tensor
    keepDims = true;
    unsigned int axisShape5[] = { 2 };
    std::set<unsigned int> axisData5 = { 1, 2 };
    TensorInfo axisTensorInfo5(1, &axisShape5[0], DataType::Signed32);

    TensorInfo outputTensorInfo5;

    CalculateReducedOutputTensoInfo(inputTensorInfo, axisTensorInfo5, axisData5,
    keepDims, outputTensorInfo5);

    BOOST_ASSERT(outputTensorInfo5.GetNumDimensions() == 3);
    BOOST_ASSERT(outputTensorInfo5.GetShape()[0] == 2);
    BOOST_ASSERT(outputTensorInfo5.GetShape()[1] == 1);
    BOOST_ASSERT(outputTensorInfo5.GetShape()[2] == 1);

}

BOOST_AUTO_TEST_SUITE_END()

