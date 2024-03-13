//
// Copyright © 2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ScatterNdTestHelper.hpp"

#include <doctest/doctest.h>

namespace armnnDelegate
{

template <typename T>
void ScatterNd1DimTest(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 3, 1 };
    std::vector<int32_t> updatesShape = { 3 };
    std::vector<int32_t> shapeShape = { 1 };
    std::vector<int32_t> expectedOutputShape = { 5 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 1, 2 };
    std::vector<T> updatesValues = { 1, 2, 3 };
    std::vector<int32_t> shapeValue = { 5 };
    std::vector<T> expectedOutputValues =  { 1, 2, 3, 0, 0 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

template <typename T>
void ScatterNd2DimTest(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 3, 2 };
    std::vector<int32_t> updatesShape = { 3 };
    std::vector<int32_t> shapeShape = { 2 };
    std::vector<int32_t> expectedOutputShape = { 3, 3 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 0,
                                           1, 1,
                                           2, 2 };
    std::vector<T> updatesValues = { 1, 2, 3 };
    std::vector<int32_t> shapeValue = { 3, 3 };
    std::vector<T> expectedOutputValues =  { 1, 0, 0,
                                             0, 2, 0,
                                             0, 0, 3 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

template <typename T>
void ScatterNd2Dim1Outter1InnerTest(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 2, 1 };
    std::vector<int32_t> updatesShape = { 2, 3 };
    std::vector<int32_t> shapeShape = { 2 };
    std::vector<int32_t> expectedOutputShape = { 3, 3 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 1 };
    std::vector<T> updatesValues = { 1, 1, 1,
                                           1, 1, 1 };
    std::vector<int32_t> shapeValue = { 3, 3 };
    std::vector<T> expectedOutputValues =  { 1, 1, 1,
                                            1, 1, 1,
                                            0, 0, 0 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

template <typename T>
void ScatterNd3DimTest(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 3, 3 };
    std::vector<int32_t> updatesShape = { 3 };
    std::vector<int32_t> shapeShape = { 3 };
    std::vector<int32_t> expectedOutputShape = { 3, 3, 3 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 0, 0,
                                           1, 1, 1,
                                           2, 2, 2 };
    std::vector<T> updatesValues = { 1, 2, 3 };
    std::vector<int32_t> shapeValue = { 3, 3, 3 };
    std::vector<T> expectedOutputValues =  { 1, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 2, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 0, 0,
                                             0, 0, 3 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

template <typename T>
void ScatterNd3Dim1Outter2InnerTest(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 2, 1 };
    std::vector<int32_t> updatesShape = { 2, 3, 3 };
    std::vector<int32_t> shapeShape = { 3 };
    std::vector<int32_t> expectedOutputShape = { 3, 3, 3 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 1 };
    std::vector<T> updatesValues = { 1, 1, 1,
                                    1, 1, 1,
                                    1, 1, 1,

                                    2, 2, 2,
                                    2, 2, 2,
                                    2, 2, 2 };
    std::vector<int32_t> shapeValue = { 3, 3, 3 };
    std::vector<T> expectedOutputValues =  { 1, 1, 1,
                                             1, 1, 1,
                                             1, 1, 1,

                                             2, 2, 2,
                                             2, 2, 2,
                                             2, 2, 2,

                                             0, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

template <typename T>
void ScatterNd3Dim2Outter1InnerTest(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 2, 2 };
    std::vector<int32_t> updatesShape = { 2, 3 };
    std::vector<int32_t> shapeShape = { 3 };
    std::vector<int32_t> expectedOutputShape = { 3, 3, 3 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 0,
                                           1, 1 };
    std::vector<T> updatesValues = { 1, 1, 1,
                                     2, 2, 2 };
    std::vector<int32_t> shapeValue = { 3, 3, 3 };
    std::vector<T> expectedOutputValues =  { 1, 1, 1,
                                             0, 0, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             2, 2, 2,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

template <typename T>
void ScatterNdDim4(tflite::TensorType tensorType, const std::vector<armnn::BackendId>& backends = {})
{
    // Set shapes
    std::vector<int32_t> indicesShape = { 3, 4 };
    std::vector<int32_t> updatesShape = { 3 };
    std::vector<int32_t> shapeShape = { 4 };
    std::vector<int32_t> expectedOutputShape = { 2, 3, 3, 3 };

    // Set Values
    std::vector<int32_t> indicesValues = { 0, 0, 0, 0,
                                           0, 1, 1, 1,
                                           1, 1, 1, 1 };
    std::vector<T> updatesValues = { 1, 2, 3 };
    std::vector<int32_t> shapeValue = { 2, 3, 3, 3 };
    std::vector<T> expectedOutputValues =  { 1, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 2, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 3, 0,
                                             0, 0, 0,

                                             0, 0, 0,
                                             0, 0, 0,
                                             0, 0, 0 };

    ScatterNdTestImpl<T>(tensorType,
                         indicesShape,
                         indicesValues,
                         updatesShape,
                         updatesValues,
                         shapeShape,
                         shapeValue,
                         expectedOutputShape,
                         expectedOutputValues,
                         backends);
}

TEST_SUITE("ScatterNdDelegateTests")
{

TEST_CASE ("ScatterNd_1Dim_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd1DimTest<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_1Dim_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd1DimTest<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_1Dim_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd1DimTest<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_1Dim_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd1DimTest<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("ScatterNd_2Dim_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd2DimTest<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_2Dim_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd2DimTest<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_2Dim_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd2DimTest<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_2Dim_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd2DimTest<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("ScatterNd_2Dim_1Outter_1Inner_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd2Dim1Outter1InnerTest<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_2Dim_1Outter_1Inner_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd2Dim1Outter1InnerTest<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_2Dim_1Outter_1Inner_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd2Dim1Outter1InnerTest<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_2Dim_1Outter_1Inner_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd2Dim1Outter1InnerTest<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("ScatterNd_3Dim_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd3DimTest<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_3Dim_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd3DimTest<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_3Dim_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd3DimTest<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_3Dim_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd3DimTest<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("ScatterNd_3Dim_1Outter_2Inner_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd3Dim1Outter2InnerTest<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_3Dim_1Outter_2Inner_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd3Dim1Outter2InnerTest<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_3Dim_1Outter_2Inner_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd3Dim1Outter2InnerTest<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_3Dim_1Outter_2Inner_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd3Dim1Outter2InnerTest<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("ScatterNd_3Dim_2Outter_1Inner_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd3Dim2Outter1InnerTest<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_3Dim_2Outter_1Inner_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNd3Dim2Outter1InnerTest<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_3Dim_2Outter_1Inner_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd3Dim2Outter1InnerTest<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_3Dim_2Outter_1Inner_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNd3Dim2Outter1InnerTest<uint8_t>(tflite::TensorType_UINT8, backends);
}

TEST_CASE ("ScatterNd_4Dim_FP32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNdDim4<float>(tflite::TensorType_FLOAT32, backends);
}

TEST_CASE ("ScatterNd_4Dim_INT32_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef, armnn::Compute::GpuAcc };
    ScatterNdDim4<int32_t>(tflite::TensorType_INT32, backends);
}

TEST_CASE ("ScatterNd_4Dim_INT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNdDim4<int8_t>(tflite::TensorType_INT8, backends);
}

TEST_CASE ("ScatterNd_4Dim_UINT8_Test")
{
    std::vector<armnn::BackendId> backends = { armnn::Compute::CpuRef };
    ScatterNdDim4<uint8_t>(tflite::TensorType_UINT8, backends);
}

} // TEST_SUITE("ScatterNdDelegateTests")

} // namespace armnnDelegate