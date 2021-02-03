//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Deprecated.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/NetworkFwd.hpp>
#include <armnn/Optional.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

namespace armnn
{
class ILayerVisitor
{
protected:
    ILayerVisitor() {}
    virtual ~ILayerVisitor() {}

public:
    /// Function an absolute layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitElementwiseUnaryLayer instead")
    virtual void VisitAbsLayer(const IConnectableLayer* layer,
                               const char* name = nullptr) = 0;

    /// Function that an activation layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param activationDescriptor - ActivationDescriptor to configure the activation.
    /// @param name - Optional name for the layer.
    virtual void VisitActivationLayer(const IConnectableLayer* layer,
                                      const ActivationDescriptor& activationDescriptor,
                                      const char* name = nullptr) = 0;

    /// Function that an addition layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitAdditionLayer(const IConnectableLayer* layer,
                                    const char* name = nullptr) = 0;

    /// Function that an arg min max layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param argMinMaxDescriptor - ArgMinMaxDescriptor to configure the activation.
    /// @param name - Optional name for the layer.
    virtual void VisitArgMinMaxLayer(const IConnectableLayer* layer,
                                     const ArgMinMaxDescriptor& argMinMaxDescriptor,
                                     const char* name = nullptr) = 0;

    /// Function that a batch normalization layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param mean - Pre-calculated mean for each channel.
    /// @param variance - Pre-calculated variance for each channel.
    /// @param beta - Per-channel additive factor.
    /// @param gamma - Per-channel multiplicative factor.
    /// @param name - Optional name for the layer.
    virtual void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                              const BatchNormalizationDescriptor& desc,
                                              const ConstTensor& mean,
                                              const ConstTensor& variance,
                                              const ConstTensor& beta,
                                              const ConstTensor& gamma,
                                              const char* name = nullptr) = 0;

    /// Function that a batch to space ND layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param batchToSpaceNdDescriptor - Description of the layer.
    /// @param name - Optional name for the layer.
    virtual void VisitBatchToSpaceNdLayer(const IConnectableLayer* layer,
                                          const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                          const char* name = nullptr) = 0;

    /// Function a Comparison layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param comparisonDescriptor - Description of the layer.
    /// @param name - Optional name for the layer.
    virtual void VisitComparisonLayer(const IConnectableLayer* layer,
                                      const ComparisonDescriptor& comparisonDescriptor,
                                      const char* name = nullptr) = 0;

    /// Function that a concat layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param concatDescriptor - ConcatDescriptor (synonym for OriginsDescriptor) to configure the concatenation
    ///                           process. Number of Views must be equal to the number of inputs, and their order
    ///                           must match - e.g. first view corresponds to the first input, second view to the
    ///                           second input, etc....
    /// @param name - Optional name for the layer.
    virtual void VisitConcatLayer(const IConnectableLayer* layer,
                                  const OriginsDescriptor& concatDescriptor,
                                  const char* name = nullptr)
    {
        // default implementation to ease transition while MergerLayer is being deprecated
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        VisitMergerLayer(layer, concatDescriptor, name);
        ARMNN_NO_DEPRECATE_WARN_END
    }

    /// Function a layer with no inputs and a single output, which always corresponds to
    /// the passed in constant tensor should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param input - Tensor to be provided as the only output of the layer. The layer will maintain
    ///                its own copy of the tensor data, meaning the memory referenced by @a input can
    ///                be freed or reused after this function is called.
    /// @param name - Optional name for the layer.
    virtual void VisitConstantLayer(const IConnectableLayer* layer,
                                    const ConstTensor& input,
                                    const char* name = nullptr) = 0;

    /// Function that a 2D convolution layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param convolution2dDescriptor - Description of the 2D convolution layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data. If specified, must match the output tensor shape.
    /// @param name - Optional name for the layer.
    virtual void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                         const Convolution2dDescriptor& convolution2dDescriptor,
                                         const ConstTensor& weights,
                                         const Optional<ConstTensor>& biases,
                                         const char* name = nullptr) = 0;

    /// Function a depth to space layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param depthToSpaceDescriptor - Parameters for the depth to space operation.
    /// @param name - Optional name for the layer.
    virtual void VisitDepthToSpaceLayer(const IConnectableLayer* layer,
                                        const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                        const char* name = nullptr) = 0;

    /// Function that a 2D depthwise convolution layer with biases should call back to when its
    /// Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param convolution2dDescriptor - Description of the 2D depthwise convolution layer.
    /// @param weights - Tensor for the weights. Expected format: [channelMultiplier, inputChannels, height, width].
    /// @param biases - Optional tensor for the bias data. If specified, must match the output tensor shape.
    /// @param name - Optional name for the layer.
    virtual void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                                  const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
                                                  const ConstTensor& weights,
                                                  const Optional<ConstTensor>& biases,
                                                  const char* name = nullptr) = 0;

    /// Function that a Dequantize layer should call back to when its
    /// Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitDequantizeLayer(const IConnectableLayer* layer,
                                      const char* name = nullptr) = 0;

    /// Function that a Detection PostProcess layer should call back to when its
    /// Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param descriptor - Description of the Detection PostProcess layer.
    /// @param anchors - Tensor for the anchors.
    /// @param name - Optional name for the layer.
    virtual void VisitDetectionPostProcessLayer(const IConnectableLayer* layer,
                                                const DetectionPostProcessDescriptor& descriptor,
                                                const ConstTensor& anchors,
                                                const char* name = nullptr) = 0;

    /// Function a division layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitDivisionLayer(const IConnectableLayer* layer,
                                    const char* name = nullptr) = 0;

    /// Function a ElementwiseUnary layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param elementwiseUnaryDescriptor - Description of the layer.
    /// @param name - Optional name for the layer.
    virtual void VisitElementwiseUnaryLayer(const IConnectableLayer* layer,
                                            const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                            const char* name = nullptr) = 0;

    /// Function an Equal layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitComparisonLayer instead")
    virtual void VisitEqualLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) = 0;

    /// Function a fill layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param fillDescriptor - Description of the layer
    /// @param name - Optional name for the layer.
    virtual void VisitFillLayer(const IConnectableLayer* layer,
                                const FillDescriptor& fillDescriptor,
                                const char* name = nullptr) = 0;

    /// Function a floor layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitFloorLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) = 0;

    /// Function that a fully connected layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param fullyConnectedDescriptor - Description of the fully connected layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data.
    /// @param name - Optional name for the layer.
    virtual void VisitFullyConnectedLayer(const IConnectableLayer* layer,
                                          const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                          const ConstTensor& weights,
                                          const Optional<ConstTensor>& biases,
                                          const char* name = nullptr) = 0;

    /// Function a Gather layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitGatherLayer with descriptor instead")
    virtual void VisitGatherLayer(const IConnectableLayer* layer,
                                  const char* name = nullptr) = 0;

    /// Function a Gather layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param gatherDescriptor - Parameters for the gather operation.
    /// @param name - Optional name for the layer.
    virtual void VisitGatherLayer(const IConnectableLayer* layer,
                                  const GatherDescriptor& gatherDescriptor,
                                  const char* name = nullptr) = 0;

    /// Function a Greater layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitComparisonLayer instead")
    virtual void VisitGreaterLayer(const IConnectableLayer* layer,
                                   const char* name = nullptr) = 0;

    /// Function that an InputLayer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param id - User generated id to uniquely identify a particular input. The same id needs to be specified
    ///             when passing the inputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    virtual void VisitInputLayer(const IConnectableLayer* layer,
                                 LayerBindingId id,
                                 const char* name = nullptr) = 0;

    /// Function that an instance normalization layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param desc - Parameters for the instance normalization operation.
    /// @param name - Optional name for the layer.
    virtual void VisitInstanceNormalizationLayer(const IConnectableLayer* layer,
                                                 const InstanceNormalizationDescriptor& desc,
                                                 const char* name = nullptr) = 0;

    /// Function that an L2 normalization layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked. Normalization is performed along dimension 1, but requires a 4d input.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param desc - Parameters for the L2 normalization operation.
    /// @param name - Optional name for the layer.
    virtual void VisitL2NormalizationLayer(const IConnectableLayer* layer,
                                           const L2NormalizationDescriptor& desc,
                                           const char* name = nullptr) = 0;

    /// Function that a log softmax layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param logSoftmaxDescriptor - LogSoftmaxDescriptor to configure the log softmax.
    /// @param name - Optional name for the layer.
    virtual void VisitLogSoftmaxLayer(const IConnectableLayer* layer,
                                      const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                      const char* name = nullptr) = 0;

    /// Function that a logical binary layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param logicalBinaryDescriptor - LogicalBinaryDescriptor to configure the logical unary layer.
    /// @param name - Optional name for the layer.
    virtual void VisitLogicalBinaryLayer(const IConnectableLayer* layer,
                                         const LogicalBinaryDescriptor& logicalBinaryDescriptor,
                                         const char* name = nullptr) = 0;

    /// Function an Lstm layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param descriptor - Parameters controlling the operation of the Lstm operation.
    /// @param params - The weights and biases for the LSTM cell.
    /// @param name - Optional name for the layer.
    virtual void VisitLstmLayer(const IConnectableLayer* layer,
                                const LstmDescriptor& descriptor,
                                const LstmInputParams& params,
                                const char* name = nullptr) = 0;

    /// Function a Maximum layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitMaximumLayer(const IConnectableLayer* layer,
                                   const char* name = nullptr) = 0;

    /// Function a Mean layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param meanDescriptor - Parameters for the mean operation.
    /// @param name - Optional name for the layer.
    virtual void VisitMeanLayer(const IConnectableLayer* layer,
                                const MeanDescriptor& meanDescriptor,
                                const char* name = nullptr) = 0;

    /// Function that a merge layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitMergeLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) = 0;

    /// Function that a merger layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param mergerDescriptor - MergerDescriptor (synonym for OriginsDescriptor) to configure the concatenation
    ///                           process. Number of Views must be equal to the number of inputs, and their order
    ///                           must match - e.g. first view corresponds to the first input, second view to the
    ///                           second input, etc....
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitConcatLayer instead")
    virtual void VisitMergerLayer(const IConnectableLayer* layer,
                                  const MergerDescriptor& mergerDescriptor,
                                  const char* name = nullptr) = 0;

    /// Function a Minimum layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitMinimumLayer(const IConnectableLayer* layer,
                                   const char* name = nullptr) = 0;

    /// Function that a multiplication layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitMultiplicationLayer(const IConnectableLayer* layer,
                                          const char* name = nullptr) = 0;

    /// Function that a normalization layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param normalizationDescriptor - NormalizationDescriptor to configure the normalization.
    /// @param name - Optional name for the layer.
    virtual void VisitNormalizationLayer(const IConnectableLayer* layer,
                                         const NormalizationDescriptor& normalizationDescriptor,
                                         const char* name = nullptr) = 0;

    /// Function an output layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param id - User generated id to uniquely identify a particular output. The same id needs to be specified
    /// when passing the outputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    virtual void VisitOutputLayer(const IConnectableLayer* layer,
                                  LayerBindingId id,
                                  const char* name = nullptr) = 0;

    /// Function a pad layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param paddings - n by 2 tensor, where n is the rank of the input tensor,
    ///                   such that paddings[i,0] indicates the amount of padding to add in front of dimension i, and
    ///                   paddings[i,1] indicates the amount of padding to add after the end of dimension i
    /// @param name - Optional name for the layer.
    virtual void VisitPadLayer(const IConnectableLayer* layer,
                               const PadDescriptor& padDescriptor,
                               const char* name = nullptr) = 0;

    /// Function that a permute layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param permuteDescriptor - PermuteDescriptor to configure the permute.
    /// @param name - Optional name for the layer.
    virtual void VisitPermuteLayer(const IConnectableLayer* layer,
                                   const PermuteDescriptor& permuteDescriptor,
                                   const char* name = nullptr) = 0;

    /// Function that a pooling layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param pooling2dDescriptor - Pooling2dDescriptor to configure the pooling.
    /// @param name - Optional name for the layer.
    virtual void VisitPooling2dLayer(const IConnectableLayer* layer,
                                     const Pooling2dDescriptor& pooling2dDescriptor,
                                     const char* name = nullptr) = 0;

    /// Function that a PReLU activation layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitPreluLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) = 0;

    /// Function a quantize layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitQuantizeLayer(const IConnectableLayer* layer,
                                    const char* name = nullptr) = 0;

    /// Function a QLstm layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param descriptor - Parameters controlling the operation of the QLstm operation.
    /// @param params - The weights and biases for the layer
    /// @param name - Optional name for the layer.
    virtual void VisitQLstmLayer(const IConnectableLayer* layer,
                                 const QLstmDescriptor& descriptor,
                                 const LstmInputParams& params,
                                 const char* name = nullptr) = 0;

    /// Function a QuantizedLstm layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param params - The weights and biases for the Quantized LSTM cell
    /// @param name - Optional name for the layer.
    virtual void VisitQuantizedLstmLayer(const IConnectableLayer* layer,
                                         const QuantizedLstmInputParams& params,
                                         const char* name = nullptr) = 0;

    /// Function a rank layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitRankLayer(const IConnectableLayer* layer,
                                const char* name = nullptr) = 0;

    /// Function that a reduce layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param ReduceDescriptor - Parameters for the reduce max operation.
    /// @param name - Optional name for the layer.
    virtual void VisitReduceLayer(const IConnectableLayer* layer,
                                  const ReduceDescriptor& reduceDescriptor,
                                  const char* name = nullptr) = 0;

    /// Function a reshape layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param reshapeDescriptor - Parameters for the reshape operation.
    /// @param name - Optional name for the layer.
    virtual void VisitReshapeLayer(const IConnectableLayer* layer,
                                   const ReshapeDescriptor& reshapeDescriptor,
                                   const char* name = nullptr) = 0;

    /// Function that a resize bilinear layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param resizeDesc - Parameters for the resize operation.
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitResizeLayer instead")
    virtual void VisitResizeBilinearLayer(const IConnectableLayer* layer,
                                          const ResizeBilinearDescriptor& resizeDesc,
                                          const char* name = nullptr) = 0;

    /// Function that a resize layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param resizeDescriptor - Parameters for the resize operation.
    /// @param name - Optional name for the layer.
    virtual void VisitResizeLayer(const IConnectableLayer* layer,
                                  const ResizeDescriptor& resizeDescriptor,
                                  const char* name = nullptr) = 0;

    /// Function a Reciprocal of square root layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    ARMNN_DEPRECATED_MSG("Use VisitElementwiseUnaryLayer instead")
    virtual void VisitRsqrtLayer(const IConnectableLayer* layer,
                                 const char* name = nullptr) = 0;

    /// Function that a slice layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param sliceDescriptor - SliceDescriptor to configure the slice operation.
    /// @param name - Optional name for the layer.
    virtual void VisitSliceLayer(const IConnectableLayer* layer,
                                 const SliceDescriptor& sliceDescriptor,
                                 const char* name = nullptr) = 0;


    /// Function that a softmax layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param softmaxDescriptor - SoftmaxDescriptor to configure the softmax.
    /// @param name - Optional name for the layer.
    virtual void VisitSoftmaxLayer(const IConnectableLayer* layer,
                                   const SoftmaxDescriptor& softmaxDescriptor,
                                   const char* name = nullptr) = 0;

    /// Function a space to batch layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param spaceToBatchNdDescriptor - Parameters for the space to batch operation.
    /// @param name - Optional name for the layer.
    virtual void VisitSpaceToBatchNdLayer(const IConnectableLayer* layer,
                                          const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                          const char* name = nullptr) = 0;

    /// Function a space to depth layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param spaceToDepthDescriptor - Parameters for the space to depth operation.
    /// @param name - Optional name for the layer.
    virtual void VisitSpaceToDepthLayer(const IConnectableLayer* layer,
                                        const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                        const char* name = nullptr) = 0;

    /// Function that a splitter layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param splitterDescriptor - ViewsDescriptor to configure the splitting process.
    ///                             Number of Views must be equal to the number of outputs,
    ///                             and their order must match - e.g. first view corresponds to
    ///                             the first output, second view to the second output, etc....
    /// @param name - Optional name for the layer.
    virtual void VisitSplitterLayer(const IConnectableLayer* layer,
                                    const ViewsDescriptor& splitterDescriptor,
                                    const char* name = nullptr) = 0;

    /// Function a stack layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param stackDescriptor - Parameters for the stack operation.
    /// @param name - Optional name for the layer.
    virtual void VisitStackLayer(const IConnectableLayer* layer,
                                 const StackDescriptor& stackDescriptor,
                                 const char* name = nullptr) = 0;

    /// Function a StandInLayer should call back to when its Accept(ILaterVisitor&) function is invoked
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param standInDescriptor - Parameters for the stand-in layer.
    /// @param name - Optional name for the layer.
    virtual void VisitStandInLayer(const IConnectableLayer* layer,
                                   const StandInDescriptor& standInDescriptor,
                                   const char* name = nullptr) = 0;

    /// Function a strided slice layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param stridedSliceDescriptor - Parameters for the strided slice operation.
    /// @param name - Optional name for the layer.
    virtual void VisitStridedSliceLayer(const IConnectableLayer* layer,
                                        const StridedSliceDescriptor& stridedSliceDescriptor,
                                        const char* name = nullptr) = 0;

    /// Function a subtraction layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitSubtractionLayer(const IConnectableLayer* layer,
                                       const char* name = nullptr) = 0;

    /// Function a switch layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param name - Optional name for the layer.
    virtual void VisitSwitchLayer(const IConnectableLayer* layer,
                                  const char* name = nullptr) = 0;

    /// Function that a 2D transpose convolution layer should call back to when its Accept(ILayerVisitor&)
    /// function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param descriptor - Description of the 2D transpose convolution layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data.
    /// @param name - Optional name for the layer.
    virtual void VisitTransposeConvolution2dLayer(const IConnectableLayer* layer,
                                                  const TransposeConvolution2dDescriptor& descriptor,
                                                  const ConstTensor& weights,
                                                  const Optional<ConstTensor>& biases,
                                                  const char* name = nullptr) = 0;

    /// Function that a transpose  layer should call back to when its Accept(ILayerVisitor&) function is invoked.
    /// @param layer - pointer to the layer which is calling back to this visit function.
    /// @param transposeDescriptor - TransposeDescriptor to configure the transpose.
    /// @param name - Optional name for the layer.
    virtual void VisitTransposeLayer(const IConnectableLayer* layer,
                                     const TransposeDescriptor& transposeDescriptor,
                                     const char* name = nullptr) = 0;

    virtual void StartVisit() {}
    virtual void FinishVisit() {}

};
} // namespace armnn
