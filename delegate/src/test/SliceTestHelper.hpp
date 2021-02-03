//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "TestUtils.hpp"

#include <armnn_delegate.hpp>
#include <armnn/DescriptorsFwd.hpp>

#include <flatbuffers/flatbuffers.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <doctest/doctest.h>

#include <string>

namespace
{

struct StridedSliceParams
{
    StridedSliceParams(std::vector<int32_t>& inputTensorShape,
                       std::vector<int32_t>& beginTensorData,
                       std::vector<int32_t>& endTensorData,
                       std::vector<int32_t>& strideTensorData,
                       std::vector<int32_t>& outputTensorShape,
                       armnn::StridedSliceDescriptor& descriptor)
        : m_InputTensorShape(inputTensorShape),
          m_BeginTensorData(beginTensorData),
          m_EndTensorData(endTensorData),
          m_StrideTensorData(strideTensorData),
          m_OutputTensorShape(outputTensorShape),
          m_Descriptor (descriptor) {}

    std::vector<int32_t> m_InputTensorShape;
    std::vector<int32_t> m_BeginTensorData;
    std::vector<int32_t> m_EndTensorData;
    std::vector<int32_t> m_StrideTensorData;
    std::vector<int32_t> m_OutputTensorShape;
    armnn::StridedSliceDescriptor m_Descriptor;
};

std::vector<char> CreateSliceTfLiteModel(tflite::TensorType tensorType,
                                         const std::vector<int32_t>& inputTensorShape,
                                         const std::vector<int32_t>& beginTensorData,
                                         const std::vector<int32_t>& endTensorData,
                                         const std::vector<int32_t>& strideTensorData,
                                         const std::vector<int32_t>& beginTensorShape,
                                         const std::vector<int32_t>& endTensorShape,
                                         const std::vector<int32_t>& strideTensorShape,
                                         const std::vector<int32_t>& outputTensorShape,
                                         const int32_t beginMask,
                                         const int32_t endMask,
                                         const int32_t ellipsisMask,
                                         const int32_t newAxisMask,
                                         const int32_t ShrinkAxisMask,
                                         const armnn::DataLayout& dataLayout)
{
    using namespace tflite;
    flatbuffers::FlatBufferBuilder flatBufferBuilder;

    std::array<flatbuffers::Offset<tflite::Buffer>, 4> buffers;
    buffers[0] = CreateBuffer(flatBufferBuilder, flatBufferBuilder.CreateVector({}));
    buffers[1] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(beginTensorData.data()),
                                                             sizeof(int32_t) * beginTensorData.size()));
    buffers[2] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(endTensorData.data()),
                                                             sizeof(int32_t) * endTensorData.size()));
    buffers[3] = CreateBuffer(flatBufferBuilder,
                              flatBufferBuilder.CreateVector(reinterpret_cast<const uint8_t*>(strideTensorData.data()),
                                                             sizeof(int32_t) * strideTensorData.size()));

    std::array<flatbuffers::Offset<Tensor>, 5> tensors;
    tensors[0] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(inputTensorShape.data(),
                                                                      inputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("input"));
    tensors[1] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(beginTensorShape.data(),
                                                                      beginTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              1,
                              flatBufferBuilder.CreateString("begin_tensor"));
    tensors[2] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(endTensorShape.data(),
                                                                      endTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              2,
                              flatBufferBuilder.CreateString("end_tensor"));
    tensors[3] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(strideTensorShape.data(),
                                                                      strideTensorShape.size()),
                              ::tflite::TensorType_INT32,
                              3,
                              flatBufferBuilder.CreateString("stride_tensor"));
    tensors[4] = CreateTensor(flatBufferBuilder,
                              flatBufferBuilder.CreateVector<int32_t>(outputTensorShape.data(),
                                                                      outputTensorShape.size()),
                              tensorType,
                              0,
                              flatBufferBuilder.CreateString("output"));


    // create operator
    tflite::BuiltinOptions operatorBuiltinOptionsType = tflite::BuiltinOptions_StridedSliceOptions;
    flatbuffers::Offset<void> operatorBuiltinOptions = CreateStridedSliceOptions(flatBufferBuilder,
                                                                                 beginMask,
                                                                                 endMask,
                                                                                 ellipsisMask,
                                                                                 newAxisMask,
                                                                                 ShrinkAxisMask).Union();

    const std::vector<int> operatorInputs{ 0, 1, 2, 3 };
    const std::vector<int> operatorOutputs{ 4 };
    flatbuffers::Offset <Operator> sliceOperator =
            CreateOperator(flatBufferBuilder,
                           0,
                           flatBufferBuilder.CreateVector<int32_t>(operatorInputs.data(), operatorInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(operatorOutputs.data(), operatorOutputs.size()),
                           operatorBuiltinOptionsType,
                           operatorBuiltinOptions);

    const std::vector<int> subgraphInputs{ 0, 1, 2, 3 };
    const std::vector<int> subgraphOutputs{ 4 };
    flatbuffers::Offset <SubGraph> subgraph =
            CreateSubGraph(flatBufferBuilder,
                           flatBufferBuilder.CreateVector(tensors.data(), tensors.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphInputs.data(), subgraphInputs.size()),
                           flatBufferBuilder.CreateVector<int32_t>(subgraphOutputs.data(), subgraphOutputs.size()),
                           flatBufferBuilder.CreateVector(&sliceOperator, 1));

    flatbuffers::Offset <flatbuffers::String> modelDescription =
            flatBufferBuilder.CreateString("ArmnnDelegate: StridedSlice Operator Model");
    flatbuffers::Offset <OperatorCode> operatorCode = CreateOperatorCode(flatBufferBuilder,
                                                                         BuiltinOperator_STRIDED_SLICE);

    flatbuffers::Offset <Model> flatbufferModel =
            CreateModel(flatBufferBuilder,
                        TFLITE_SCHEMA_VERSION,
                        flatBufferBuilder.CreateVector(&operatorCode, 1),
                        flatBufferBuilder.CreateVector(&subgraph, 1),
                        modelDescription,
                        flatBufferBuilder.CreateVector(buffers.data(), buffers.size()));

    flatBufferBuilder.Finish(flatbufferModel);

    return std::vector<char>(flatBufferBuilder.GetBufferPointer(),
                             flatBufferBuilder.GetBufferPointer() + flatBufferBuilder.GetSize());
}

template <typename T>
void StridedSliceTestImpl(std::vector<armnn::BackendId>& backends,
                          std::vector<T>& inputValues,
                          std::vector<T>& expectedOutputValues,
                          std::vector<int32_t>& beginTensorData,
                          std::vector<int32_t>& endTensorData,
                          std::vector<int32_t>& strideTensorData,
                          std::vector<int32_t>& inputTensorShape,
                          std::vector<int32_t>& beginTensorShape,
                          std::vector<int32_t>& endTensorShape,
                          std::vector<int32_t>& strideTensorShape,
                          std::vector<int32_t>& outputTensorShape,
                          const int32_t beginMask = 0,
                          const int32_t endMask = 0,
                          const int32_t ellipsisMask = 0,
                          const int32_t newAxisMask = 0,
                          const int32_t ShrinkAxisMask = 0,
                          const armnn::DataLayout& dataLayout = armnn::DataLayout::NHWC)
{
    using namespace tflite;
    std::vector<char> modelBuffer = CreateSliceTfLiteModel(
            ::tflite::TensorType_FLOAT32,
            inputTensorShape,
            beginTensorData,
            endTensorData,
            strideTensorData,
            beginTensorShape,
            endTensorShape,
            strideTensorShape,
            outputTensorShape,
            beginMask,
            endMask,
            ellipsisMask,
            newAxisMask,
            ShrinkAxisMask,
            dataLayout);

    auto tfLiteModel = GetModel(modelBuffer.data());

    // Create TfLite Interpreters
    std::unique_ptr<Interpreter> armnnDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
              (&armnnDelegate) == kTfLiteOk);
    CHECK(armnnDelegate != nullptr);
    CHECK(armnnDelegate->AllocateTensors() == kTfLiteOk);

    std::unique_ptr<Interpreter> tfLiteDelegate;
    CHECK(InterpreterBuilder(tfLiteModel, ::tflite::ops::builtin::BuiltinOpResolver())
              (&tfLiteDelegate) == kTfLiteOk);
    CHECK(tfLiteDelegate != nullptr);
    CHECK(tfLiteDelegate->AllocateTensors() == kTfLiteOk);

    // Create the ArmNN Delegate
    armnnDelegate::DelegateOptions delegateOptions(backends);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
    theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                     armnnDelegate::TfLiteArmnnDelegateDelete);
    CHECK(theArmnnDelegate != nullptr);

    // Modify armnnDelegateInterpreter to use armnnDelegate
    CHECK(armnnDelegate->ModifyGraphWithDelegate(theArmnnDelegate.get()) == kTfLiteOk);

    // Set input data
    armnnDelegate::FillInput<T>(tfLiteDelegate, 0, inputValues);
    armnnDelegate::FillInput<T>(armnnDelegate, 0, inputValues);

    // Run EnqueWorkload
    CHECK(tfLiteDelegate->Invoke() == kTfLiteOk);
    CHECK(armnnDelegate->Invoke() == kTfLiteOk);

    // Compare output data
    armnnDelegate::CompareOutputData<T>(tfLiteDelegate,
                                        armnnDelegate,
                                        outputTensorShape,
                                        expectedOutputValues);

    tfLiteDelegate.reset(nullptr);
    armnnDelegate.reset(nullptr);
} // End of StridedSlice Test

} // anonymous namespace