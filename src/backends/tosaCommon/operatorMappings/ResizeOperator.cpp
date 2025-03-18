//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Copyright © 2020, 2023 The TensorFlow Authors. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include "ResizeOperator.hpp"
#include "TosaRescaleOperatorUtils.hpp"

// This function is paraphrased from:
// tensorflow/compiler/mlir/tosa/transforms/legalize_common.cc from function convertResizeOp
// tensorflow/lite/kernels/internal/reference/resize_utils.h
TosaSerializationBasicBlock* ConvertResizeToTosaOperator(const Layer* layer,
                                                         const std::vector<const TensorInfo*>& inputs,
                                                         const std::vector<const TensorInfo*>& outputs,
                                                         const ResizeDescriptor* resizeDescriptor)
{
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE( inputs.size() == 1,
                                         "ConvertResizeToTosaOperator: Resize must have only one input." );
    ARMNN_THROW_INVALIDARG_MSG_IF_FALSE( resizeDescriptor->m_DataLayout == DataLayout::NHWC,
                                         "ConvertResizeToTosaOperator: NCHW not supported.");

    ResizeMode mode;
    if (resizeDescriptor->m_Method == ResizeMethod::NearestNeighbor)
    {
        mode = tosa::ResizeMode_NEAREST;
    }
    else if (resizeDescriptor->m_Method == ResizeMethod::Bilinear)
    {
        mode = tosa::ResizeMode_BILINEAR;
    }
    else
    {
        throw armnn::InvalidArgumentException("ConvertResizeToTosaOperator: Unsupported Resize method.");
    }

    std::string inputName = std::string("input_");
    std::string outputName = std::string("output0_");
    std::string blockName  = std::string("Op_RESIZE_block_") + GetUniqueTosaMappingID();

    // If a layer is present then the block will be used for execution, so input and output names need to be determined
    // using the previous and following layers so the graph is connected correctly. For validation this doesn't matter.
    if(layer != nullptr)
    {
        inputName  = GenerateUniqueInputName(layer->GetInputSlot(0));
        outputName = GenerateUniqueOutputName(*layer);
    }

    int32_t inputHeight = static_cast<int32_t>(inputs[0]->GetShape()[1]);
    int32_t inputWidth = static_cast<int32_t>(inputs[0]->GetShape()[2]);

    int32_t outputHeight = static_cast<int32_t>(resizeDescriptor->m_TargetHeight);
    int32_t outputWidth = static_cast<int32_t>(resizeDescriptor->m_TargetWidth);
    bool alignCorners = resizeDescriptor->m_AlignCorners;
    bool halfPixel = resizeDescriptor->m_HalfPixelCenters;

    // Go from ArmNN parameters (outputShape, halfPixel and alignedCorners)
    // to TOSA parameters (scale, offset and border)
    // Align corners sets the scaling ratio to (O - 1)/(I - 1) rather than O / I.
    auto preprocessResizeParameters = [&](int inputSize, int outputSize, int& scale_n, int& scale_d, int& offset)
    {
        // Dimension is length 1, we are just sampling from one value.
        if (inputSize == 1)
        {
            scale_n = outputSize;
            scale_d = 1;
            offset = 0;
            return;
        }

        // Apply if aligned and capable to be aligned.
        // Align corners sets the scaling ratio to (OH - 1)/(IH - 1) rather than OH / IH. Same for width.
        bool applyAligned = alignCorners && (outputSize > 1);
        scale_n = applyAligned ? (outputSize - 1) : outputSize;
        scale_d = applyAligned ? (inputSize - 1) : inputSize;

        // Simplify the scales, make sure they are even values.
        int gcd = std::gcd(scale_n, scale_d);
        scale_n = 2 * scale_n / gcd;
        scale_d = 2 * scale_d / gcd;

        // If half pixel centers then input and output sampling positions are offset by 1/2 pixel.
        offset = halfPixel ? (scale_d / 2 - scale_n / 2) : 0;

        // Reduce the scaling ratio if possible, we know scale_n and scale_d are even
        if ((offset & 1) == 0)
        {
            scale_n /= 2;
            scale_d /= 2;
            offset /= 2;
        }
    };

    int scale_y_n, scale_y_d, offset_y;
    int scale_x_n, scale_x_d, offset_x;
    preprocessResizeParameters(inputHeight, outputHeight, scale_y_n, scale_y_d, offset_y);
    preprocessResizeParameters(inputWidth, outputWidth, scale_x_n, scale_x_d, offset_x);

    int border_y = scale_y_d * (outputHeight - 1) - scale_y_n * (inputHeight - 1) + offset_y;
    int border_x = scale_x_d * (outputWidth - 1) - scale_x_n * (inputWidth - 1) + offset_x;

    // [scale_y_n, scale_y_d, scale_x_n, scale_x_d]
    std::vector<int16_t> scale = { static_cast<int16_t>(scale_y_n),
                                   static_cast<int16_t>(scale_y_d),
                                   static_cast<int16_t>(scale_x_n),
                                   static_cast<int16_t>(scale_x_d) };

    // [offset_y, offset_x]
    std::vector<int16_t> offset = { static_cast<int16_t>(offset_y),
                                    static_cast<int16_t>(offset_x) };
    // [border_y, border_x]
    std::vector<int16_t> border = { static_cast<int16_t>(border_y),
                                    static_cast<int16_t>(border_x) };

    auto isInt16Range = [](int x)
    {
        return (x <= std::numeric_limits<int16_t>::max()) && (x >= std::numeric_limits<int16_t>::min());
    };

    if (inputs[0]->IsQuantized())
    {
        // It isn't commonly seen these numbers aren't fit within 16 bits, and won't match TFLite reference.
        if (!isInt16Range(scale_y_n) || !isInt16Range(scale_y_d) ||
            !isInt16Range(scale_x_n) || !isInt16Range(scale_x_d) ||
            !isInt16Range(offset_y) || !isInt16Range(offset_x) ||
            !isInt16Range(border_y) || !isInt16Range(border_x))
        {
            throw armnn::Exception("ConvertResizeToTosaOperator: stride or offset out of 16 bit range");
        }
    }

    TosaResizeAttribute resizeAttribute(scale, offset, border, mode);

    std::vector<TosaSerializationTensor*> tensors;

    DType inputDType = ArmNNToDType(inputs[0]->GetDataType());
    DType outputDType = ArmNNToDType(outputs[0]->GetDataType());

    if(inputs[0]->GetDataType() == DataType::QSymmS16 && mode == tosa::ResizeMode_BILINEAR)
    {
        throw armnn::Exception("ConvertResizeToTosaOperator(): Bilinear INT16 is not yet implemented.");
    }

    if (inputs[0]->GetDataType() == DataType::Signed32 && mode == tosa::ResizeMode_BILINEAR)
    {
        throw armnn::Exception("ConvertResizeToTosaOperator(): Bilinear INT32 is not supported.");
    }

    // Only add input tensors if connected layer is an input layer.
    // As intermediate or constant tensors will be created separately.
    // There also can't be duplicate tensor.
    if (inputName.find("input_") != std::string::npos)
    {
        std::vector<int32_t> inputShape = GetTosaTensorShape(inputs[0]->GetShape());

        tensors.push_back(new TosaSerializationTensor(inputName, inputShape, inputDType, {}));
    }

    std::vector<int32_t> outputShape = GetTosaTensorShape(outputs[0]->GetShape());

    if (mode == tosa::ResizeMode_BILINEAR &&
        inputDType == DType::DType_INT8 &&
        outputDType == DType::DType_INT8)
    {
        std::string inoutResizeToRescale = std::string("inout_resize2rescale_bilinear_") + GetUniqueTosaMappingID();

        //For this scenario the resize output TOSA tensor type is a scaled INT32 value. Need to
        //convert to unscaled INT8
        tensors.push_back(new TosaSerializationTensor(inoutResizeToRescale, outputShape, DType::DType_INT32, {}));

        auto* resizeOp = new TosaSerializationOperator(Op_RESIZE,
                                                       Attribute_ResizeAttribute,
                                                       &resizeAttribute,
                                                       {inputName},
                                                       {inoutResizeToRescale});

        tensors.push_back(new TosaSerializationTensor(outputName, outputShape, outputDType, {}));

        //As per TOSA spec INT32 output is scaled by scale_y_n * scale_x_n for bilinear resize
        double scale_bi { 1. / static_cast<double>(scale_y_n * scale_x_n) };

        TosaSerializationOperator* rescaleOp {nullptr};

        CreateRescaleTosaOperator(inoutResizeToRescale,
                                  outputName,
                                  scale_bi,
                                  0,
                                  0,
                                  false,
                                  false,
                                  true,
                                  true,
                                  &rescaleOp);

        return new TosaSerializationBasicBlock(blockName,                  // name
                                               mainName,                   // region name
                                               { resizeOp, rescaleOp },    // operators
                                               tensors,                    // tensors
                                               { inputName },              // inputs
                                               { outputName });            // outputs
    }
    else
    {
        tensors.push_back(new TosaSerializationTensor(outputName, outputShape, outputDType, {}));

        auto* op = new TosaSerializationOperator(Op_RESIZE,
                                                 Attribute_ResizeAttribute,
                                                 &resizeAttribute,
                                                 {inputName},
                                                 {outputName});

        // operatorInputNames/operatorOutputNames ends up being the same as
        // blockInputNames/blockOutputNames for one-to-one ArmNN to TOSA mappings
        return new TosaSerializationBasicBlock(blockName, // name
                                               mainName, // region name
                                               {op}, // operators
                                               tensors, // tensors
                                               {inputName}, // inputs
                                               {outputName}); // outputs
    }
}