//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <TestUtils.hpp>

#include <Graph.hpp>
#include <layers/ArgMinMaxLayer.hpp>
#include <layers/BatchToSpaceNdLayer.hpp>
#include <layers/SpaceToDepthLayer.hpp>
#include <layers/PreluLayer.hpp>
#include <layers/StackLayer.hpp>

#include <doctest/doctest.h>

void ArgMinMaxInferOutputShapeImpl(const armnn::ArgMinMaxDescriptor       descriptor,
                                   const std::vector<armnn::TensorShape>& inputShapes,
                                   std::vector<armnn::TensorShape>&       outputShapes)
{
    armnn::Graph graph;
    auto argMinMaxLayer = graph.AddLayer<armnn::ArgMinMaxLayer>(descriptor, "argMinMax");
    outputShapes = argMinMaxLayer->InferOutputShapes(inputShapes);
}

void ArgMinMaxInferOutputShape4dTest()
{
    armnn::Graph graph;
    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Axis = 2;

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 1, 3, 2, 4 }
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(ArgMinMaxInferOutputShapeImpl(descriptor, inputShapes, outputShapes));

    armnn::TensorShape expectedOutputShape( { 1, 3, 4 } );
    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShape);
}

void ArgMinMaxInferOutputShape3dTest()
{
    armnn::Graph graph;
    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Axis = 0;

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 1, 3, 2 }
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(ArgMinMaxInferOutputShapeImpl(descriptor, inputShapes, outputShapes));

    armnn::TensorShape expectedOutputShape( { 3, 2 } );
    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShape);
}

void ArgMinMaxInferOutputShape2dTest()
{
    armnn::Graph graph;
    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Axis = 1;

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 3, 2 }
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(ArgMinMaxInferOutputShapeImpl(descriptor, inputShapes, outputShapes));

    armnn::TensorShape expectedOutputShape( { 3 } );
    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShape);
}

void ArgMinMaxInferOutputShape1dTest()
{
    armnn::Graph graph;
    armnn::ArgMinMaxDescriptor descriptor;
    descriptor.m_Axis = 0;

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 5 }
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(ArgMinMaxInferOutputShapeImpl(descriptor, inputShapes, outputShapes));

    armnn::TensorShape expectedOutputShape( { 1 } );
    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShape);
}

void BatchToSpaceInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::BatchToSpaceNdDescriptor descriptor;
    descriptor.m_BlockShape = {2, 2};
    descriptor.m_Crops = {{0, 0}, {2, 0}};
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::BatchToSpaceNdLayer* const batchToSpaceLayer =
        graph.AddLayer<armnn::BatchToSpaceNdLayer>(descriptor, "batchToSpace");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> theDimSizes = {8, 1, 3, 1};
    armnn::TensorShape shape(4, theDimSizes.data());
    shapes.push_back(shape);

    const std::vector<unsigned int> expectedDimSizes = {2, 2, 4, 1};
    armnn::TensorShape expectedShape(4, expectedDimSizes.data());

    CHECK(expectedShape == batchToSpaceLayer->InferOutputShapes(shapes).at(0));
}

void SpaceToDepthInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::SpaceToDepthDescriptor descriptor;
    descriptor.m_BlockSize  = 2;
    descriptor.m_DataLayout = armnn::DataLayout::NHWC;

    armnn::SpaceToDepthLayer* const spaceToDepthLayer =
        graph.AddLayer<armnn::SpaceToDepthLayer>(descriptor, "spaceToDepth");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> dimSizes{ 1, 16, 8, 3 };
    armnn::TensorShape shape(4, dimSizes.data());
    shapes.push_back(shape);

    const std::vector<unsigned int> expectedDimSizes{ 1, 8, 4, 12 };
    armnn::TensorShape expectedShape(4, expectedDimSizes.data());

    CHECK(expectedShape == spaceToDepthLayer->InferOutputShapes(shapes).at(0));
}

void PreluInferOutputShapeImpl(const std::vector<armnn::TensorShape>& inputShapes,
                               std::vector<armnn::TensorShape>&       outputShapes)
{
    armnn::Graph graph;
    armnn::PreluLayer* const preluLayer = graph.AddLayer<armnn::PreluLayer>("prelu");
    outputShapes = preluLayer->InferOutputShapes(inputShapes);
}

void PreluInferOutputShapeSameDimsTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 5, 1, 1, 7 }, // Input shape
        { 5, 4, 3, 1 }  // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 4, 3, 7 }  // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

void PreluInferOutputShapeInputBiggerTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 4, 8 }, // Input shape
        { 5, 4, 1 }     // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 4, 5, 4, 8 } // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

void PreluInferOutputShapeAlphaBiggerTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 2 },   // Input shape
        { 5, 4, 3, 1 } // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 4, 3, 2 } // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShapes[0]);
}

void PreluInferOutputShapeNoMatchTest()
{
    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 1, 2 },   // Input shape
        { 5, 4, 3, 1 } // Alpha shape
    };

    const std::vector<armnn::TensorShape> expectedOutputShapes
    {
        { 5, 7, 3, 2 } // Output shape
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(PreluInferOutputShapeImpl(inputShapes, outputShapes));

    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] != expectedOutputShapes[0]);
}

void CreatePreluLayerHelper(armnn::Graph& graph,
                            const armnn::TensorShape& inputShape,
                            const armnn::TensorShape& alphaShape,
                            const armnn::TensorShape& outputShape)
{
    // Creates the PReLU layer
    armnn::Layer* const preluLayer = graph.AddLayer<armnn::PreluLayer>("prelu");

    // Creates extra layers
    armnn::Layer* const input  = graph.AddLayer<armnn::InputLayer> (0, "input");
    armnn::Layer* const alpha  = graph.AddLayer<armnn::InputLayer> (1, "alpha");
    armnn::Layer* const output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    // Connects up
    armnn::TensorInfo inputTensorInfo (inputShape,  armnn::DataType::Float32);
    armnn::TensorInfo alphaTensorInfo (alphaShape,  armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
    Connect(input, preluLayer,  inputTensorInfo,  0, 0);
    Connect(alpha, preluLayer,  alphaTensorInfo,  0, 1);
    Connect(preluLayer, output, outputTensorInfo, 0, 0);
}

void PreluValidateTensorShapesFromInputsMatchTest()
{
    armnn::Graph graph;

    // Creates the PReLU layer
    CreatePreluLayerHelper(graph, { 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 4, 3, 2 });

    // Graph::InferTensorInfos calls Layer::ValidateTensorShapesFromInputs
    CHECK_NOTHROW(graph.InferTensorInfos());
}

void PreluValidateTensorShapesFromInputsNoMatchTest()
{
    armnn::Graph graph;

    // Creates the PReLU layer
    CreatePreluLayerHelper(graph, { 1, 4, 1, 2 }, { 5, 4, 3, 1 }, { 5, 7, 3, 2 });

    // Graph::InferTensorInfos calls Layer::ValidateTensorShapesFromInputs
    CHECK_THROWS_AS(graph.InferTensorInfos(), armnn::LayerValidationException);
}

void StackInferOutputShapeImpl(const armnn::StackDescriptor           descriptor,
                               const std::vector<armnn::TensorShape>& inputShapes,
                               std::vector<armnn::TensorShape>&       outputShapes)
{
    armnn::Graph graph;
    armnn::StackLayer* const stackLayer = graph.AddLayer<armnn::StackLayer>(descriptor, "stack");
    outputShapes = stackLayer->InferOutputShapes(inputShapes);
}

void StackInferOutputShapeFromInputsMatchTest()
{
    armnn::Graph graph;

    armnn::StackDescriptor descriptor;
    descriptor.m_Axis = 1;
    descriptor.m_NumInputs = 3;
    descriptor.m_InputShape = armnn::TensorShape
    (
        { 4, 2 }  // Defined input shape
    );

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 2 }, // Actual input shapes
        { 4, 2 },
        { 4, 2 }
    };

    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(StackInferOutputShapeImpl(descriptor, inputShapes, outputShapes));

    armnn::TensorShape expectedOutputShape
    (
        { 4, 3, 2 }
    );
    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShape);
}

void StackInferOutputShapeFromInputsNoMatchTest()
{
    armnn::Graph graph;

    armnn::StackDescriptor descriptor;
    descriptor.m_Axis = 1;
    descriptor.m_NumInputs = 3;
    descriptor.m_InputShape = armnn::TensorShape
    (
        { 4, 2 }  // Defined input shape
    );

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 4, 2 }, // Actual input shapes
        { 4, 5 }, // Incorrectly shaped input tensor
        { 4, 2 }
    };

    // Output shape is inferred from the descriptor, so should still be correct despite mismatching input shapes
    std::vector<armnn::TensorShape> outputShapes;
    CHECK_NOTHROW(StackInferOutputShapeImpl(descriptor, inputShapes, outputShapes));

    armnn::TensorShape expectedOutputShape
    (
        { 4, 3, 2 }
    );
    CHECK(outputShapes.size() == 1);
    CHECK(outputShapes[0] == expectedOutputShape);
}

void CreateStackLayerHelper(armnn::Graph& graph,
                            const armnn::StackDescriptor& descriptor,
                            const std::vector<armnn::TensorShape>& inputShapes,
                            const armnn::TensorShape& outputShape)
{
    // Creates the Stack layer
    armnn::Layer* const stackLayer = graph.AddLayer<armnn::StackLayer>(descriptor, "stack");

    // Creates extra layers
    std::vector<armnn::Layer*> inputs;
    for (unsigned int i=0; i<inputShapes.size(); ++i)
    {
        inputs.push_back(graph.AddLayer<armnn::InputLayer>(static_cast<int>(i), "input"));
    }
    armnn::Layer* const output = graph.AddLayer<armnn::OutputLayer>(0, "output");

    // Connects up
    std::vector<armnn::TensorInfo> inputTensorInfos;
    for (unsigned int i=0; i<inputs.size(); ++i)
    {
        inputTensorInfos.push_back(armnn::TensorInfo(inputShapes[i], armnn::DataType::Float32));
    }
    armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);

    for (unsigned int i=0; i<inputs.size(); ++i)
    {
        Connect(inputs[i], stackLayer, inputTensorInfos[i], 0, i);
    }
    Connect(stackLayer, output, outputTensorInfo, 0, 0);
}

void StackValidateTensorShapesFromInputsMatchTest()
{
    armnn::Graph graph;

    armnn::StackDescriptor descriptor;
    descriptor.m_Axis = 0;
    descriptor.m_NumInputs = 3;
    descriptor.m_InputShape = armnn::TensorShape
    (
        { 2, 5 }  // Defined input shape
    );

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 2, 5 }, // Actual input shapes
        { 2, 5 },
        { 2, 5 }
    };

    // Creates the Stack layer
    CreateStackLayerHelper(graph, descriptor, inputShapes, { 3, 2, 5 });

    // Graph::InferTensorInfos calls Layer::ValidateTensorShapesFromInputs
    CHECK_NOTHROW(graph.InferTensorInfos());
}

void StackValidateTensorShapesFromInputsNoMatchTest()
{
    armnn::Graph graph;

    armnn::StackDescriptor descriptor;
    descriptor.m_Axis = 0;
    descriptor.m_NumInputs = 3;
    descriptor.m_InputShape = armnn::TensorShape
    (
        { 2, 5 }  // Defined input shape
    );

    const std::vector<armnn::TensorShape> inputShapes
    {
        { 2, 5 }, // Actual input shapes
        { 2, 2 }, // Incorrectly shaped input tensor
        { 2, 5 }
    };

    // Creates the Stack layer
    CreateStackLayerHelper(graph, descriptor, inputShapes, { 3, 2, 5 });

    // Graph::InferTensorInfos calls Layer::ValidateTensorShapesFromInputs
    CHECK_THROWS_AS(graph.InferTensorInfos(), armnn::LayerValidationException);
}

void Convolution2dInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::Convolution2dDescriptor descriptor;
    descriptor.m_DilationX = 2;
    descriptor.m_DilationY = 2;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_StrideX = 3;
    descriptor.m_StrideY = 3;
    descriptor.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::Convolution2dLayer* const convolution2dLayer =
            graph.AddLayer<armnn::Convolution2dLayer>(descriptor, "convolution2d");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> inputSize = {1, 2, 10, 10};
    armnn::TensorShape inputShape(4, inputSize.data());
    shapes.push_back(inputShape);

    const std::vector<unsigned int> filterSize = { 1, 2, 2, 2};
    armnn::TensorShape filterShape(4, filterSize.data());
    shapes.push_back(filterShape);

    const std::vector<unsigned int> expectedOutputSizes = {1, 1, 4, 4};
    armnn::TensorShape expectedOutputShape(4, expectedOutputSizes.data());

    CHECK(expectedOutputShape == convolution2dLayer->InferOutputShapes(shapes).at(0));
}

void Convolution3dInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::Convolution3dDescriptor descriptor;
    descriptor.m_DilationX = 1;
    descriptor.m_DilationY = 1;
    descriptor.m_DilationZ = 1;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 1;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 2;
    descriptor.m_DataLayout = armnn::DataLayout::NDHWC;

    armnn::Convolution3dLayer* const convolution3dLayer =
            graph.AddLayer<armnn::Convolution3dLayer>(descriptor, "convolution3d");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> inputSize = {1, 5, 5, 5, 1};
    armnn::TensorShape inputShape(5, inputSize.data());
    shapes.push_back(inputShape);

    const std::vector<unsigned int> filterSize = {3, 3, 3, 1, 1 };
    armnn::TensorShape filterShape(5, filterSize.data());
    shapes.push_back(filterShape);

    const std::vector<unsigned int> expectedOutputSizes = {1, 3, 3, 3, 1};
    armnn::TensorShape expectedOutputShape(5, expectedOutputSizes.data());

    CHECK(expectedOutputShape == convolution3dLayer->InferOutputShapes(shapes).at(0));
}

void TransposeConvolution2dInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::TransposeConvolution2dDescriptor descriptor;
    descriptor.m_PadTop = 0;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadLeft = 0;
    descriptor.m_PadRight = 1;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::TransposeConvolution2dLayer* const transposeConvolution2dLayer =
            graph.AddLayer<armnn::TransposeConvolution2dLayer>(descriptor, "TransposeConvolution2d");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> inputSize = {1, 2, 3, 3};
    armnn::TensorShape inputShape(4, inputSize.data());
    shapes.push_back(inputShape);

    const std::vector<unsigned int> filterSize = { 1, 2, 3, 3};
    armnn::TensorShape filterShape(4, filterSize.data());
    shapes.push_back(filterShape);

    const std::vector<unsigned int> expectedOutputSizes = {1, 1, 6, 6};
    armnn::TensorShape expectedOutputShape(4, expectedOutputSizes.data());

    CHECK(expectedOutputShape == transposeConvolution2dLayer->InferOutputShapes(shapes).at(0));
}

void DepthwiseConvolution2dInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::DepthwiseConvolution2dDescriptor descriptor;
    descriptor.m_DilationX = 3;
    descriptor.m_DilationY = 3;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 2;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 2;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_DataLayout = armnn::DataLayout::NCHW;

    armnn::DepthwiseConvolution2dLayer* const depthwiseConvolution2dLayer =
            graph.AddLayer<armnn::DepthwiseConvolution2dLayer>(descriptor, "DepthwiseConvolution2d");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> inputSize = {1, 2, 10, 10};
    armnn::TensorShape inputShape(4, inputSize.data());
    shapes.push_back(inputShape);

    const std::vector<unsigned int> filterSize = { 1, 3, 3, 2 };
    armnn::TensorShape filterShape(4, filterSize.data());
    shapes.push_back(filterShape);

    const std::vector<unsigned int> expectedOutputSizes = {1, 2, 4, 4};
    armnn::TensorShape expectedOutputShape(4, expectedOutputSizes.data());

    CHECK(expectedOutputShape == depthwiseConvolution2dLayer->InferOutputShapes(shapes).at(0));
}

void Pooling3dInferOutputShapeTest()
{
    armnn::Graph graph;

    armnn::Pooling3dDescriptor descriptor;
    descriptor.m_PoolType = armnn::PoolingAlgorithm::Max;
    descriptor.m_PoolDepth = 2;
    descriptor.m_PoolHeight = 2;
    descriptor.m_PoolWidth = 2;
    descriptor.m_PadTop = 1;
    descriptor.m_PadBottom = 1;
    descriptor.m_PadLeft = 1;
    descriptor.m_PadRight = 1;
    descriptor.m_PadFront = 1;
    descriptor.m_PadBack = 1;
    descriptor.m_StrideX = 2;
    descriptor.m_StrideY = 2;
    descriptor.m_StrideZ = 2;
    descriptor.m_DataLayout = armnn::DataLayout::NDHWC;

    armnn::Pooling3dLayer* const pooling3dLayer =
            graph.AddLayer<armnn::Pooling3dLayer>(descriptor, "pooling3d");

    std::vector<armnn::TensorShape> shapes;
    const std::vector<unsigned int> inputSize = {1, 4, 4, 4, 1};
    armnn::TensorShape inputShape(5, inputSize.data());
    shapes.push_back(inputShape);

    const std::vector<unsigned int> expectedOutputSizes = {1, 3, 3, 3, 1};
    armnn::TensorShape expectedOutputShape(5, expectedOutputSizes.data());

    CHECK(expectedOutputShape == pooling3dLayer->InferOutputShapes(shapes).at(0));
}

// QLstm
void QLstmInferOutputShapeImpl(const armnn::QLstmDescriptor descriptor,
                               const std::vector<armnn::TensorShape>& inputShapes,
                               std::vector<armnn::TensorShape>& outputShapes)
{
    armnn::Graph graph;
    armnn::QLstmLayer* const qLstmLayer = graph.AddLayer<armnn::QLstmLayer>(descriptor, "qLstm");
    outputShapes = qLstmLayer->InferOutputShapes(inputShapes);
}

void QLstmInferOutputShapeTest()
{
    armnn::QLstmDescriptor descriptor;
    descriptor.m_PeepholeEnabled = true;
    descriptor.m_CifgEnabled = false;
    descriptor.m_ProjectionEnabled = false;

    // Input shapes
    const std::vector<unsigned int> inputShape{ 2, 5 };
    const std::vector<unsigned int> previousOutputInShape{ 2, 4 };
    const std::vector<unsigned int> previousCellStateInShape{ 2, 4 };

    armnn::TensorShape inputTensorShape(2, inputShape.data());
    armnn::TensorShape previousOutputInTensorShape(2, previousOutputInShape.data());
    armnn::TensorShape previousCellStateInTensorShape(2, previousCellStateInShape.data());

    std::vector<armnn::TensorShape> inShapes
    {
        inputTensorShape,
        previousOutputInTensorShape,
        previousCellStateInTensorShape
    };

    // Output shapes
    const std::vector<unsigned int> outputStateOutShape{ 2, 4 };
    const std::vector<unsigned int> cellStateOutShape{ 2, 4 };
    const std::vector<unsigned int> outputShape{ 2, 4 };
    armnn::TensorShape outputStateOutTensorShape(2, outputShape.data());
    armnn::TensorShape cellStateOutTensorShape(2, cellStateOutShape.data());
    armnn::TensorShape outputTensorShape(2, outputShape.data());

    std::vector<armnn::TensorShape> expectedOutShapes
    {
        outputStateOutTensorShape,
        cellStateOutTensorShape,
        outputTensorShape
    };

    std::vector<armnn::TensorShape> actualOutShapes;
    CHECK_NOTHROW(QLstmInferOutputShapeImpl(descriptor, inShapes, actualOutShapes));

    CHECK(actualOutShapes.size() == 3);
    CHECK(expectedOutShapes[0] == actualOutShapes[0]);
    CHECK(expectedOutShapes[1] == actualOutShapes[1]);
    CHECK(expectedOutShapes[2] == actualOutShapes[2]);
}

// QuantizedLstm
void QuantizedLstmInferOutputShapeImpl(const std::vector<armnn::TensorShape>& inputShapes,
                                       std::vector<armnn::TensorShape>& outputShapes)
{
    armnn::Graph graph;
    armnn::QuantizedLstmLayer* const quantizedLstmLayer = graph.AddLayer<armnn::QuantizedLstmLayer>("quantizedLstm");
    outputShapes = quantizedLstmLayer->InferOutputShapes(inputShapes);
}

void QuantizedLstmInferOutputShapeTest()
{
    // Input shapes
    const std::vector<unsigned int> inputShape{ 2, 5 };
    const std::vector<unsigned int> previousCellStateInShape{ 2, 10 };
    const std::vector<unsigned int> previousOutputInShape{ 2, 10 };
    armnn::TensorShape inputTensorShape(2, inputShape.data());
    armnn::TensorShape previousCellStateInTensorShape(2, previousCellStateInShape.data());
    armnn::TensorShape previousOutputInTensorShape(2, previousOutputInShape.data());

    std::vector<armnn::TensorShape> inShapes
    {
        inputTensorShape,
        previousCellStateInTensorShape,
        previousOutputInTensorShape
    };

    // Output shapes
    const std::vector<unsigned int> cellStateOutShape{ 2, 10 };
    const std::vector<unsigned int> outputShape{ 2, 10 };
    armnn::TensorShape cellStateOutTensorShape(2, cellStateOutShape.data());
    armnn::TensorShape outputTensorShape(2, outputShape.data());

    std::vector<armnn::TensorShape> expectedOutShapes
    {
        cellStateOutTensorShape,
        outputTensorShape
    };

    std::vector<armnn::TensorShape> actualOutShapes;
    CHECK_NOTHROW(QuantizedLstmInferOutputShapeImpl(inShapes, actualOutShapes));

    CHECK(actualOutShapes.size() == 2);
    CHECK(expectedOutShapes[0] == actualOutShapes[0]);
    CHECK(expectedOutShapes[1] == actualOutShapes[1]);
}