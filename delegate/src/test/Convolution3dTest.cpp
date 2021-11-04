//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ConvolutionTestHelper.hpp"

#include <armnn_delegate.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include <doctest/doctest.h>

namespace armnnDelegate
{

// Conv3d is currently only supports Float32 inputs, filter, bias and outputs in TFLite.
// Conv3d is only correctly supported for external delegates from TF Lite v2.6, as there was a breaking bug in v2.5.
#if defined(ARMNN_POST_TFLITE_2_5)

// Create a vector from 0 to size divided to create smaller floating point values.
template <typename T>
std::vector<T> CreateFloatData(int32_t size, float divisor)
{
    std::vector<float> data;
    for (int32_t i = 0; i < size; ++i)
    {
        float value = static_cast<float>(i);
        data.push_back(value/divisor);
    }
    return data;
}

void Conv3DWithBiasesSimpleWithPaddingFp32Test(std::vector<armnn::BackendId>& backends)
{
    // Set input data
    std::vector<int32_t> inputShape { 1, 2, 2, 2, 1 };
    std::vector<int32_t> filterShape { 2, 2, 2, 1, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 2, 2, 2, 1 };

    static std::vector<float> inputValues =
    {
        1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f
    };

    std::vector<float> filterValues =
    {
        2.f,1.f, 1.f,0.f, 0.f,1.f, 1.f,1.f
    };

    std::vector<float> biasValues = { 5.f };

    std::vector<float> expectedOutputValues =
    {
       33.f, 21.f, 23.f, 13.f, 28.f, 25.f, 27.f, 21.f
    };

    Convolution3dTest<float>(tflite::BuiltinOperator_CONV_3D,
                             ::tflite::TensorType_FLOAT32,
                             { 1, 1, 1 }, // strideX, strideY, strideZ
                             { 1, 1, 1 }, // dilationX, dilationY, dilationZ
                             tflite::Padding_SAME,
                             tflite::ActivationFunctionType_NONE,
                             backends,
                             inputShape,
                             filterShape,
                             outputShape,
                             inputValues,
                             filterValues,
                             expectedOutputValues,
                             biasShape,
                             biasValues);
}

void Conv3DWithBiasesStridesFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 3, 10, 10, 1 };
    std::vector<int32_t> filterShape { 3, 5, 5, 1, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 1, 3, 3, 1 };

    std::vector<float> inputValues = CreateFloatData<float>(300, 1.0f);

    std::vector<float> filterValues =
    {
        1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f,
        1.f, 1.f, 1.f, 1.f, 1.f,

        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,
        0.f, 0.f, 0.f, 0.f, 0.f,

        2.f, 2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f, 2.f,
        2.f, 2.f, 2.f, 2.f, 2.f
    };

    std::vector<float> biasValues = { 10.f };

    std::vector<float> expectedOutputValues =
    {
        11660.f, 11810.f, 11960.f,

        13160.f, 13310.f, 13460.f,

        14660.f, 14810.f, 14960.f
    };

    Convolution3dTest<float>(tflite::BuiltinOperator_CONV_3D,
                             ::tflite::TensorType_FLOAT32,
                             { 2, 2, 2 }, // strideX, strideY, strideZ
                             { 1, 1, 1 }, // dilationX, dilationY, dilationZ
                             tflite::Padding_VALID,
                             tflite::ActivationFunctionType_NONE,
                             backends,
                             inputShape,
                             filterShape,
                             outputShape,
                             inputValues,
                             filterValues,
                             expectedOutputValues,
                             biasShape,
                             biasValues);
}


void Conv3DWithBiasesDilationFp32Test(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 5, 5, 5, 2 };
    std::vector<int32_t> filterShape { 2, 2, 2, 2, 2 };
    std::vector<int32_t> biasShape { 2 };
    std::vector<int32_t> outputShape { 1, 2, 2, 2, 2 };

    std::vector<float> inputValues = CreateFloatData<float>(250, 1.0f);

    std::vector<float> filterValues =
    {
        -1.f, -1.f,  -1.f, -1.f,  -1.f, -1.f,  -1.f, -1.f,  -1.f, -1.f,  -1.f,  1.f,   1.f,  1.f,  -1.f, -1.f,
         1.f,  1.f,  -1.f,  1.f,  -1.f,  1.f,  -1.f,  1.f,  -1.f, -1.f,  -1.f,  1.f,  -1.f,  1.f,  -1.f,  1.f,
    };

    std::vector<float> biasValues = { 0.f, 2.f };

    // Since the dilation rate is 3 this will dilate the kernel to be 4x4,
    // therefore the output will be 2x2
    std::vector<float> expectedOutputValues =
    {
        -1124.f, 976.f,
        -1148.f, 980.f,

        -1244.f, 996.f,
        -1268.f, 1000.f,

        -1724.f, 1076.f,
        -1748.f, 1080.f,

        -1844.f, 1096.f,
        -1868.f, 1100.f
    };

    Convolution3dTest<float>(tflite::BuiltinOperator_CONV_3D,
                             ::tflite::TensorType_FLOAT32,
                             { 1, 1, 1 }, // strideX, strideY, strideZ
                             { 3, 3, 3 }, // dilationX, dilationY, dilationZ
                             tflite::Padding_VALID,
                             tflite::ActivationFunctionType_NONE,
                             backends,
                             inputShape,
                             filterShape,
                             outputShape,
                             inputValues,
                             filterValues,
                             expectedOutputValues,
                             biasShape,
                             biasValues);
}

void Conv3DFp32SmallTest(std::vector<armnn::BackendId>& backends)
{
    std::vector<int32_t> inputShape { 1, 3, 10, 10, 1 };
    std::vector<int32_t> filterShape { 3, 3, 3, 1, 1 };
    std::vector<int32_t> biasShape { 1 };
    std::vector<int32_t> outputShape { 1, 1, 4, 4, 1 };

    std::vector<float> inputValues = CreateFloatData<float>(300, 100.0f);

    std::vector<float> filterValues =
    {
         0.125977f,  0.150391f,  0.101562f,
         0.0585938f, 0.0864258f, 0.043457f,
         0.034668f,  0.0322266f, 0.0385742f,

         0.125977f,  0.150391f, -0.101562f,
        -0.0585938f,-0.0864258f,-0.043457f,
        -0.0104630f, 0.0154114f, 0.0013768f,

         0.0344238f, 0.035644f,  0.0495605f,
         0.0683594f, 0.099121f, -0.0461426f,
        -0.0996094f,-0.126953f, -0.043457f,
    };

    std::vector<float> biasValues = { 0 };

    std::vector<float> expectedOutputValues =
    {
        -0.08156067f, -0.06891209f, -0.05589598f, -0.04310101f,
         0.04584253f,  0.05855697f,  0.07129729f,  0.08325434f,
         0.17304349f,  0.18521416f,  0.19818866f,  0.21096253f,
         0.29965734f,  0.312698f,    0.32547557f,  0.33818722f
    };

    Convolution3dTest<float>(tflite::BuiltinOperator_CONV_3D,
                             ::tflite::TensorType_FLOAT32,
                             { 2, 2, 2 }, // strideX, strideY, strideZ
                             { 1, 1, 1 }, // dilationX, dilationY, dilationZ
                             tflite::Padding_VALID,
                             tflite::ActivationFunctionType_NONE,
                             backends,
                             inputShape,
                             filterShape,
                             outputShape,
                             inputValues,
                             filterValues,
                             expectedOutputValues,
                             biasShape,
                             biasValues);
}

TEST_SUITE("Convolution3dTest_CpuRefTests")
{

TEST_CASE ("Conv3DWithBiasesSimpleWithPadding_Fp32_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    Conv3DWithBiasesSimpleWithPaddingFp32Test(backends);
}

TEST_CASE ("Conv3DWithBiasesStrides_Fp32_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    Conv3DWithBiasesStridesFp32Test(backends);
}

TEST_CASE ("Conv3DWithBiasesDilation_Fp32_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    Conv3DWithBiasesDilationFp32Test(backends);
}

TEST_CASE ("Conv3DFp32Small_Fp32_CpuRef_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuRef};
    Conv3DFp32SmallTest(backends);
}

} //End of TEST_SUITE("Convolution3dTest_CpuRefTests")

TEST_SUITE("Convolution3dTest_CpuAccTests")
{

TEST_CASE ("Conv3DWithBiasesSimpleWithPadding_Fp32_CpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    Conv3DWithBiasesSimpleWithPaddingFp32Test(backends);
}

TEST_CASE ("Conv3DWithBiasesStrides_Fp32_CpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    Conv3DWithBiasesStridesFp32Test(backends);
}

TEST_CASE ("Conv3DFp32Small_Fp32_CpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::CpuAcc};
    Conv3DFp32SmallTest(backends);
}

} //End of TEST_SUITE("Convolution3dTest_CpuAccTests")

TEST_SUITE("Convolution3dTest_GpuAccTests")
{

TEST_CASE ("Conv3DWithBiasesSimpleWithPadding_Fp32_GpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    Conv3DWithBiasesSimpleWithPaddingFp32Test(backends);
}

TEST_CASE ("Conv3DWithBiasesStrides_Fp32_GpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    Conv3DWithBiasesStridesFp32Test(backends);
}

TEST_CASE ("Conv3DFp32Small_Fp32_GpuAcc_Test")
{
    std::vector <armnn::BackendId> backends = {armnn::Compute::GpuAcc};
    Conv3DFp32SmallTest(backends);
}

} //End of TEST_SUITE("Convolution3dTest_GpuAccTests")

#endif

} // namespace armnnDelegate