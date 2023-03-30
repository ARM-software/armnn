//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <DelegateTestInterpreterUtils.hpp>

#include <armnn_delegate.hpp>

#include <armnn/BackendId.hpp>
#include <armnn/Exceptions.hpp>

#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/kernels/kernel_util.h>
#include <tensorflow/lite/kernels/custom_ops_register.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/c/c_api_internal.h>

namespace delegateTestInterpreter
{

class DelegateTestInterpreter
{
public:
    /// Create TfLite Interpreter only
    DelegateTestInterpreter(std::vector<char>& modelBuffer, const std::string& customOp = "")
    {
        TfLiteModel* model = delegateTestInterpreter::CreateTfLiteModel(modelBuffer);

        TfLiteInterpreterOptions* options = delegateTestInterpreter::CreateTfLiteInterpreterOptions();
        if (!customOp.empty())
        {
            options->mutable_op_resolver = delegateTestInterpreter::GenerateCustomOpResolver(customOp);
        }

        m_TfLiteInterpreter = TfLiteInterpreterCreate(model, options);
        m_TfLiteDelegate = nullptr;

        // The options and model can be deleted after the interpreter is created.
        TfLiteInterpreterOptionsDelete(options);
        TfLiteModelDelete(model);
    }

    /// Create Interpreter with default Arm NN Classic/Opaque Delegate applied
    DelegateTestInterpreter(std::vector<char>& model,
                            const std::vector<armnn::BackendId>& backends,
                            const std::string& customOp = "",
                            bool disableFallback = true);

    /// Create Interpreter with Arm NN Classic/Opaque Delegate applied and DelegateOptions
    DelegateTestInterpreter(std::vector<char>& model,
                            const armnnDelegate::DelegateOptions& delegateOptions,
                            const std::string& customOp = "");

    /// Allocate the TfLiteTensors within the graph.
    /// This must be called before FillInputTensor(values, index) and Invoke().
    TfLiteStatus AllocateTensors()
    {
        return TfLiteInterpreterAllocateTensors(m_TfLiteInterpreter);
    }

    /// Copy a buffer of values into an input tensor at a given index.
    template<typename T>
    TfLiteStatus FillInputTensor(std::vector<T>& inputValues, int index)
    {
        TfLiteTensor* inputTensor = delegateTestInterpreter::GetInputTensorFromInterpreter(m_TfLiteInterpreter, index);
        return delegateTestInterpreter::CopyFromBufferToTensor(inputTensor, inputValues);
    }

    /// Copy a boolean buffer of values into an input tensor at a given index.
    /// Boolean types get converted to a bit representation in a vector.
    /// vector.data() returns a void pointer instead of a pointer to bool, so the tensor needs to be accessed directly.
    TfLiteStatus FillInputTensor(std::vector<bool>& inputValues, int index)
    {
        TfLiteTensor* inputTensor = delegateTestInterpreter::GetInputTensorFromInterpreter(m_TfLiteInterpreter, index);
        if(inputTensor->type != kTfLiteBool)
        {
            throw armnn::Exception("Input tensor at the given index is not of bool type: " + std::to_string(index));
        }

        // Make sure there is enough bytes allocated to copy into.
        if(inputTensor->bytes < inputValues.size() * sizeof(bool))
        {
            throw armnn::Exception("Input tensor has not been allocated to match number of input values.");
        }

        for (unsigned int i = 0; i < inputValues.size(); ++i)
        {
            inputTensor->data.b[i] = inputValues[i];
        }

        return kTfLiteOk;
    }

    /// Run the interpreter either on TFLite Runtime or Arm NN Delegate.
    /// AllocateTensors() must be called before Invoke().
    TfLiteStatus Invoke()
    {
        return TfLiteInterpreterInvoke(m_TfLiteInterpreter);
    }

    /// Return a buffer of values from the output tensor at a given index.
    /// This must be called after Invoke().
    template<typename T>
    std::vector<T> GetOutputResult(int index)
    {
        const TfLiteTensor* outputTensor =
                delegateTestInterpreter::GetOutputTensorFromInterpreter(m_TfLiteInterpreter, index);

        int64_t n = tflite::NumElements(outputTensor);
        std::vector<T> output;
        output.resize(n);

        TfLiteStatus status = TfLiteTensorCopyToBuffer(outputTensor, output.data(), output.size() * sizeof(T));
        if(status != kTfLiteOk)
        {
            throw armnn::Exception("An error occurred when copying output buffer.");
        }

        return output;
    }

    /// Return a buffer of values from the output tensor at a given index. This must be called after Invoke().
    /// Boolean types get converted to a bit representation in a vector.
    /// vector.data() returns a void pointer instead of a pointer to bool, so the tensor needs to be accessed directly.
    std::vector<bool> GetOutputResult(int index)
    {
        const TfLiteTensor* outputTensor =
                delegateTestInterpreter::GetOutputTensorFromInterpreter(m_TfLiteInterpreter, index);
        if(outputTensor->type != kTfLiteBool)
        {
            throw armnn::Exception("Output tensor at the given index is not of bool type: " + std::to_string(index));
        }

        int64_t n = tflite::NumElements(outputTensor);
        std::vector<bool> output(n, false);
        output.reserve(n);

        for (unsigned int i = 0; i < output.size(); ++i)
        {
            output[i] = outputTensor->data.b[i];
        }
        return output;
    }

    /// Return a buffer of dimensions from the output tensor at a given index.
    std::vector<int32_t> GetOutputShape(int index)
    {
        const TfLiteTensor* outputTensor =
                delegateTestInterpreter::GetOutputTensorFromInterpreter(m_TfLiteInterpreter, index);
        int32_t numDims = TfLiteTensorNumDims(outputTensor);

        std::vector<int32_t> dims;
        dims.reserve(numDims);

        for (int32_t i = 0; i < numDims; ++i)
        {
            dims.push_back(TfLiteTensorDim(outputTensor, i));
        }
        return dims;
    }

    /// Delete TfLiteInterpreter and the TfLiteDelegate/TfLiteOpaqueDelegate
    void Cleanup();

private:
    TfLiteInterpreter* m_TfLiteInterpreter;

    /// m_TfLiteDelegate can be TfLiteDelegate or TfLiteOpaqueDelegate
    void* m_TfLiteDelegate;
};

} // anonymous namespace