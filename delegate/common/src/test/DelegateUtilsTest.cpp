//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Tensor.hpp>
#include <doctest/doctest.h>
#include <common/src/DelegateUtils.hpp>

namespace armnn
{

TEST_SUITE("DelegateUtils_Tests")
{
    TEST_CASE("Zero_Dim_In_Input_Test_True")
    {
        unsigned int inputDimSizes[] = {0, 1, 2, 3};
        TensorInfo inputTensor = armnn::TensorInfo(4, inputDimSizes, DataType::Float32);

        CHECK(ZeroDimPresent({inputTensor}) == true);
    }

    TEST_CASE("Zero_Dim_In_Input_Test_False")
    {
        unsigned int inputDimSizes[] = {1, 2, 3, 4};
        TensorInfo inputTensor = armnn::TensorInfo(4, inputDimSizes, DataType::Float32);

        CHECK(ZeroDimPresent({inputTensor}) == false);
    }

    TEST_CASE("Zero_Dim_In_Output_Test_True")
    {
        unsigned int inputDimSizes[] = {1, 2, 3, 4};
        TensorInfo inputTensor = armnn::TensorInfo(4, inputDimSizes, DataType::Float32);

        unsigned int outputDimSizes[] = {0, 1, 2, 3};
        TensorInfo outputTensor = armnn::TensorInfo(4, outputDimSizes, DataType::Float32);

        CHECK(ZeroDimPresent({inputTensor, outputTensor}) == true);
    }

    TEST_CASE("Zero_Dim_In_Output_Test_False")
    {
        unsigned int inputDimSizes[] = {1, 2, 3, 4};
        TensorInfo inputTensor = armnn::TensorInfo(4, inputDimSizes, DataType::Float32);

        unsigned int outputDimSizes[] = {1, 2, 3, 4};
        TensorInfo outputTensor = armnn::TensorInfo(4, outputDimSizes, DataType::Float32);

        CHECK(ZeroDimPresent({inputTensor, outputTensor}) == false);
    }

    TEST_CASE("Grouped_Conv_2_Groups")
    {
        TensorShape inputShape({2, 2, 2, 2});
        TensorShape filterShape({2, 2, 2, 1});

        CHECK(IsGroupedConvolution(inputShape, filterShape, armnn::DataLayout::NHWC) == true);
    }

    TEST_CASE("Grouped_Conv_No_Groups")
    {
        TensorShape inputShape({2, 2, 2, 2});
        TensorShape filterShape({2, 2, 2, 2});

        CHECK(IsGroupedConvolution(inputShape, filterShape, armnn::DataLayout::NHWC) == false);
    }
}

}    // namespace armnn