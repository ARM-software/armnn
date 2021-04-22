//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendOptions.hpp>
#include <armnn/Deprecated.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/ILayerVisitor.hpp>
#include <armnn/IStrategy.hpp>
#include <armnn/NetworkFwd.hpp>
#include <armnn/Optional.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Types.hpp>

#include <memory>
#include <vector>

namespace armnn
{
/// @brief An input connection slot for a layer.
/// The input slot can be connected to an output slot of the preceding layer in the graph.
/// Only one connection to the input slot is allowed.
class IInputSlot
{
public:
    virtual const IOutputSlot* GetConnection() const = 0;
    virtual IOutputSlot* GetConnection() = 0;

protected:
   /// Not user deletable.
    ~IInputSlot() {}
};

/// @brief An output connection slot for a layer.
/// The output slot may be connected to 1 or more input slots of subsequent layers in the graph.
class IOutputSlot
{
public:
    virtual unsigned int GetNumConnections() const = 0;
    virtual const IInputSlot* GetConnection(unsigned int index) const = 0;
    virtual IInputSlot* GetConnection(unsigned int index) = 0;

    virtual void SetTensorInfo(const TensorInfo& tensorInfo) = 0;
    virtual const TensorInfo& GetTensorInfo() const = 0;
    virtual bool IsTensorInfoSet() const = 0;

    virtual int Connect(IInputSlot& destination) = 0;
    virtual void Disconnect(IInputSlot& slot) = 0;

    virtual unsigned int CalculateIndexOnOwner() const = 0;

    virtual LayerGuid GetOwningLayerGuid() const = 0;

protected:
    /// Not user deletable.
    ~IOutputSlot() {}
};

/// @brief Interface for a layer that is connectable to other layers via InputSlots and OutputSlots.
class IConnectableLayer
{
public:
    /// Returns the name of the layer
    virtual const char* GetName() const = 0;

    /// Returns the number of connectable input slots
    virtual unsigned int GetNumInputSlots() const = 0;

    /// Returns the number of connectable output slots
    virtual unsigned int GetNumOutputSlots() const = 0;

    /// Get a const input slot handle by slot index
    virtual const IInputSlot& GetInputSlot(unsigned int index) const = 0;

    /// Get the input slot handle by slot index
    virtual IInputSlot& GetInputSlot(unsigned int index) = 0;

    /// Get the const output slot handle by slot index
    virtual const IOutputSlot& GetOutputSlot(unsigned int index) const = 0;

    /// Get the output slot handle by slot index
    virtual IOutputSlot& GetOutputSlot(unsigned int index) = 0;

    /// Infer the shape of the output(s) based on the provided input shape(s)
    virtual std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const = 0;

    /// Returns the unique id of the layer
    virtual LayerGuid GetGuid() const = 0;

    /// Apply a visitor to this layer
    virtual void Accept(ILayerVisitor& visitor) const = 0;

    /// Apply a visitor to this layer
    virtual void ExecuteStrategy(IStrategy& strategy) const = 0;

    /// Provide a hint for the optimizer as to which backend to prefer for this layer
    virtual void BackendSelectionHint(Optional<BackendId> backend) = 0;

    /// Returns the armnn::LayerType of this layer
    virtual LayerType GetType() const = 0;

protected:
      /// Objects are not deletable via the handle
    ~IConnectableLayer() {}
};


struct OptimizerOptions
{
    OptimizerOptions()
        : m_ReduceFp32ToFp16(false)
        , m_Debug(false)
        , m_ReduceFp32ToBf16(false)
        , m_shapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly)
        , m_ImportEnabled(false)
        , m_ModelOptions()
    {}

    OptimizerOptions(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16, bool importEnabled,
                     ModelOptions modelOptions = {})
        : m_ReduceFp32ToFp16(reduceFp32ToFp16)
        , m_Debug(debug)
        , m_ReduceFp32ToBf16(reduceFp32ToBf16)
        , m_shapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly)
        , m_ImportEnabled(importEnabled)
        , m_ModelOptions(modelOptions)
    {
        if (m_ReduceFp32ToFp16 && m_ReduceFp32ToBf16)
        {
            throw InvalidArgumentException("BFloat16 and Float16 optimization cannot be enabled at the same time.");
        }
    }

    OptimizerOptions(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16 = false,
                     ShapeInferenceMethod shapeInferenceMethod = armnn::ShapeInferenceMethod::ValidateOnly,
                     bool importEnabled = false, ModelOptions modelOptions = {})
        : m_ReduceFp32ToFp16(reduceFp32ToFp16)
        , m_Debug(debug)
        , m_ReduceFp32ToBf16(reduceFp32ToBf16)
        , m_shapeInferenceMethod(shapeInferenceMethod)
        , m_ImportEnabled(importEnabled)
        , m_ModelOptions(modelOptions)
    {
        if (m_ReduceFp32ToFp16 && m_ReduceFp32ToBf16)
        {
            throw InvalidArgumentException("BFloat16 and Float16 optimization cannot be enabled at the same time.");
        }
    }

    // Reduce Fp32 data to Fp16 for faster processing
    bool m_ReduceFp32ToFp16;

    // Add debug data for easier troubleshooting
    bool m_Debug;

    // Reduce Fp32 data to Bf16 for faster processing
    bool m_ReduceFp32ToBf16;

    // Infer output size when not available
    ShapeInferenceMethod m_shapeInferenceMethod;

    // Enable Import
    bool m_ImportEnabled;

    // Enable Model Options
    ModelOptions m_ModelOptions;
};

class IWorkloadFactory;
class NetworkImpl;
using INetworkPtr = std::unique_ptr<INetwork, void(*)(INetwork* network)>;
using IOptimizedNetworkPtr = std::unique_ptr<IOptimizedNetwork, void(*)(IOptimizedNetwork* network)>;

/// Main network class which provides the interface for building up a neural network.
/// This object is subsequently required by the IRuntime::Load() method.
class INetwork
{
public:
    static INetwork* CreateRaw(NetworkOptions networkOptions = {});
    static INetworkPtr Create(NetworkOptions networkOptions = {});
    static void Destroy(INetwork* network);

    Status PrintGraph();

    /// Adds an input layer to the network.
    /// @param id - User generated id to uniquely identify a particular input. The same id needs to be specified.
    /// when passing the inputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name = nullptr);

    /// Adds an ArgMinMax layer to the network.
    /// @param desc - Parameters for the L2 normalization operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddArgMinMaxLayer(const ArgMinMaxDescriptor& desc,
                                         const char* name = nullptr);

    /// Adds a cast layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddCastLayer(const char* name = nullptr);

    /// Add a Comparison layer to the network.
    /// @param name - Optional name for the layer.
    /// @param desc - Descriptor for the comparison operation.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddComparisonLayer(const ComparisonDescriptor& comparisonDescriptor,
                                          const char* name = nullptr);

    /// Adds a concatenation layer to the network.
    /// @param concatDescriptor - ConcatDescriptor (synonym for OriginsDescriptor) to configure the concatenation
    ///                           process. Number of Views must be equal to the number of inputs, and their order
    ///                           must match - e.g. first view corresponds to the first input, second view to the
    ///                           second input, etc....
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddConcatLayer(const ConcatDescriptor& concatDescriptor,
                                      const char* name = nullptr);

    /// Adds a 2D convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D convolution layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data. If specified, must match the output tensor shape.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const Optional<ConstTensor>& biases,
                                             const char* name = nullptr);

    ARMNN_DEPRECATED_MSG("This AddConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const char* name = nullptr);

    ARMNN_DEPRECATED_MSG("This AddConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const ConstTensor& weights,
                                             const ConstTensor& biases,
                                             const char* name = nullptr);

    /// Adds a depth to space layer to the network.
    /// @param depthToSpaceDescriptor - Parameters for the depth to space operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                            const char* name = nullptr);

    /// Adds a 2D depthwise convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D depthwise convolution layer.
    /// @param weights - Tensor for the weights. Expected format: [channelMultiplier, inputChannels, height, width].
    /// @param biases Optional tensor for the bias data. If specified, must match the output tensor shape.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const Optional<ConstTensor>& biases,
        const char* name = nullptr);

    ARMNN_DEPRECATED_MSG("This AddDepthwiseConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr);

    ARMNN_DEPRECATED_MSG("This AddDepthwiseConvolution2dLayer overload is deprecated")
    IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr);

    /// Adds a Dequantize layer to the network.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDequantizeLayer(const char* name = nullptr);

    /// Adds a Detection PostProcess layer to the network.
    /// @param descriptor - Description of the Detection PostProcess layer.
    /// @param anchors - Tensor for anchors.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDetectionPostProcessLayer(
        const DetectionPostProcessDescriptor& descriptor,
        const ConstTensor& anchors,
        const char* name = nullptr);

    /// Add an ElementwiseUnary layer to the network.
    /// @param name - Optional name for the layer.
    /// @param desc - Descriptor for the elementwiseUnary operation.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                const char* name = nullptr);

    /// Add an Fill layer to the network.
    /// @param name - Optional name for the layer.
    /// @param fillDescriptor - Descriptor for the fill operation.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddFillLayer(const FillDescriptor& fillDescriptor,
                                    const char* name = nullptr);

    /// Adds a fully connected layer to the network.
    /// @param fullyConnectedDescriptor - Description of the fully connected layer.
    /// @param weights -Optional Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const Optional<ConstTensor>& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char* name = nullptr);

    /// Adds a fully connected layer to the network.
    /// @param fullyConnectedDescriptor - Description of the fully connected layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const ConstTensor& weights,
                                              const Optional<ConstTensor>& biases,
                                              const char* name = nullptr);

    ARMNN_DEPRECATED_MSG("This AddFullyConnectedLayer overload is deprecated")
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const ConstTensor& weights,
                                              const char* name = nullptr);

    ARMNN_DEPRECATED_MSG("This AddFullyConnectedLayer overload is deprecated")
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
                                              const ConstTensor& weights,
                                              const ConstTensor& biases,
                                              const char* name = nullptr);

    /// Adds a permute layer to the network.
    /// @param permuteDescriptor - PermuteDescriptor to configure the permute.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                       const char* name = nullptr);

    /// Adds a batch to space ND layer to the network.
    /// @param batchToSpaceNdDescriptor - Description of the layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                              const char* name = nullptr);

    /// Adds a pooling layer to the network.
    /// @param pooling2dDescriptor - Pooling2dDescriptor to configure the pooling.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr);

    /// Adds an activation layer to the network.
    /// @param activationDescriptor - ActivationDescriptor to configure the activation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr);

    /// Adds a normalization layer to the network.
    /// @param normalizationDescriptor - NormalizationDescriptor to configure the normalization.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr);

    /// Adds a slice layer to the network.
    /// @param sliceDescriptor - SliceDescriptor to configure the slice operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSliceLayer(const SliceDescriptor& sliceDescriptor, const char* name = nullptr);

    /// Adds a softmax layer to the network.
    /// If the data type is QAsymm8, then the output quantization parameters
    /// must have a scale of 1/256 and an offset of 0
    /// @param softmaxDescriptor - SoftmaxDescriptor to configure the softmax.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr);

    /// Adds a splitter layer to the network.
    /// @param splitterDescriptor - ViewsDescriptor to configure the splitting process.
    ///                             Number of Views must be equal to the number of outputs,
    ///                             and their order must match - e.g. first view corresponds to
    ///                             the first output, second view to the second output, etc....
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSplitterLayer(const ViewsDescriptor& splitterDescriptor,
                                        const char* name = nullptr);

    /// Adds a merge layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddMergeLayer(const char* name = nullptr);

    /// Adds a concat layer to the network.
    /// @param mergerDescriptor - MergerDescriptor (synonym for OriginsDescriptor) to configure the concatenation
    ///                           process. Number of Views must be equal to the number of inputs, and their order
    ///                           must match - e.g. first view corresponds to the first input, second view to the
    ///                           second input, etc....
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddConcatLayer instead")
    IConnectableLayer* AddMergerLayer(const MergerDescriptor& mergerDescriptor,
        const char* name = nullptr);

    /// Add absolute layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddElementwiseUnaryLayer instead")
    IConnectableLayer* AddAbsLayer(const char* name = nullptr);

    /// Adds an addition layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddAdditionLayer(const char* name = nullptr);

    /// Adds a multiplication layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr);

    /// Adds a batch normalization layer to the network.
    /// @param mean - Pre-calculated mean for each channel.
    /// @param variance - Pre-calculated variance for each channel.
    /// @param beta - Per-channel additive factor.
    /// @param gamma - Per-channel multiplicative factor.
    /// @return - Interface for configuring the layer.
    /// @param name - Optional name for the layer.
    IConnectableLayer* AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
        const ConstTensor& mean,
        const ConstTensor& variance,
        const ConstTensor& beta,
        const ConstTensor& gamma,
        const char* name = nullptr);

    /// Adds a rank layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddRankLayer(const char* name = nullptr);

    /// Adds a resize bilinear layer to the network.
    /// @param resizeDesc - Parameters for the resize operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddResizeLayer instead")
    IConnectableLayer* AddResizeBilinearLayer(const ResizeBilinearDescriptor& resizeDesc,
                                              const char* name = nullptr);

    /// Adds a resize layer to the network.
    /// @param resizeDescriptor - Parameters for the resize operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddResizeLayer(const ResizeDescriptor& resizeDescriptor,
                                      const char* name = nullptr);

    /// Adds a reduce layer to the network.
    /// @param ReduceDescriptor - Parameters for the reduce operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddReduceLayer(const ReduceDescriptor& reduceDescriptor,
                                      const char* name = nullptr);

    /// Adds an instance normalization layer to the network.
    /// @param desc - Parameters for the instance normalization operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddInstanceNormalizationLayer(const InstanceNormalizationDescriptor& desc,
                                                     const char* name = nullptr);

    /// Adds an L2 normalization layer to the network.
    /// Normalization is performed along dimension 1, but requires a 4d input.
    /// @param desc - Parameters for the L2 normalization operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                               const char* name = nullptr);

    /// Adds a log softmax layer to the network.
    /// @param logSoftmaxDescriptor - LogSoftmaxDescriptor to configure the log softmax.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddLogSoftmaxLayer(const LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                          const char* name = nullptr);

    /// Adds a layer with no inputs and a single output, which always corresponds to
    /// the passed in constant tensor.
    /// @param input - Tensor to be provided as the only output of the layer. The layer will maintain
    ///                its own copy of the tensor data, meaning the memory referenced by @a input can
    ///                be freed or reused after this function is called.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddConstantLayer(const ConstTensor& input,
                                        const char* name = nullptr);

    /// Adds a reshape layer to the network.
    /// @param reshapeDescriptor - Parameters for the reshape operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                       const char* name = nullptr);

    /// Adds a space to batch layer to the network.
    /// @param spaceToBatchNdDescriptor - Parameters for the space to batch operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                              const char* name = nullptr);

    /// Adds a space to depth layer to the network.
    /// @param spaceToDepthDescriptor - Parameters for the space to depth operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSpaceToDepthLayer(const SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                            const char* name = nullptr);

    /// Adds a floor layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddFloorLayer(const char* name = nullptr);

    /// Adds an output layer to the network.
    /// @param id - User generated id to uniquely identify a particular output. The same id needs to be specified
    /// when passing the outputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name = nullptr);

    /// Add a Lstm layer to the network
    /// @param descriptor - Parameters for the Lstm operation
    /// @param params - Weights and biases for the LSTM cell
    /// @param name - Optional name for the layer
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddLstmLayer(const LstmDescriptor& descriptor,
                                    const LstmInputParams& params,
                                    const char* name = nullptr);

    /// Adds a division layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDivisionLayer(const char* name = nullptr);

    /// Adds a subtraction layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSubtractionLayer(const char* name = nullptr);

    /// Add a Maximum layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddMaximumLayer(const char* name = nullptr);

    /// Add a Mean layer to the network.
    /// @param meanDescriptor - Parameters for the mean operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name = nullptr);

    /// Adds a fully pad layer to the network.
    /// @param paddings - n by 2 tensor, where n is the rank of the input tensor,
    ///                   such that paddings[i,0] indicates the amount of padding to add in front of dimonsion i, and
    ///                   paddings[i,1] indicates the amount of padding to add after the end of dimension i
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPadLayer(const PadDescriptor& padDescriptor,
                                           const char* name = nullptr);

    /// Add a quantize layer to the network
    ///@param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddQuantizeLayer(const char* name = nullptr);

    /// Adds a strided slice layer to the network.
    /// @param StridedSliceDescriptor - Parameters for the strided slice operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                                    const char* name = nullptr);

    /// Add a Minimum layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddMinimumLayer(const char* name = nullptr);

    /// Add a Greater layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddComparisonLayer instead")
    IConnectableLayer* AddGreaterLayer(const char* name = nullptr);

    /// Add a Equal layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddComparisonLayer instead")
    IConnectableLayer* AddEqualLayer(const char* name = nullptr);

    /// Add Reciprocal of square root layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddElementwiseUnaryLayer instead")
    IConnectableLayer* AddRsqrtLayer(const char* name = nullptr);

    /// Add Gather layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG("Use AddGatherLayer with descriptor instead")
    IConnectableLayer* AddGatherLayer(const char* name = nullptr);

    /// Add Gather layer to the network.
    /// @param descriptor - Description of the gather layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddGatherLayer(const GatherDescriptor& descriptor,
                                              const char* name = nullptr);

    /// Adds a switch layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddSwitchLayer(const char* name = nullptr);

    /// Adds a PReLU layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPreluLayer(const char* name = nullptr);

    /// Adds a 2D transpose convolution layer to the network.
    /// @param descriptor - Description of the 2D transpose convolution layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - Optional tensor for the bias data.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddTransposeConvolution2dLayer(const TransposeConvolution2dDescriptor& descriptor,
                                                              const ConstTensor& weights,
                                                              const Optional<ConstTensor>& biases,
                                                              const char* name = nullptr);

    /// Adds a transpose layer to the network.
    /// @param transposeDescriptor - TransposeDescriptor to configure the transpose.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddTransposeLayer(const TransposeDescriptor& transposeDescriptor,
                                                 const char* name = nullptr);

    /// Adds a stack layer to the network.
    /// @param descriptor - Description of the stack layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddStackLayer(const StackDescriptor& descriptor,
                                             const char* name = nullptr);

    /// Add a stand-in layer for a type unknown to the Arm NN framework.
    /// Note: Due to the nature of this layer, no validation can be performed by the framework.
    /// Furthermore, Any model containing this layer cannot make use of dynamic tensors since the
    /// tensor sizes cannot be inferred.
    /// @descriptor - Descriptor for the StandIn layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddStandInLayer(const StandInDescriptor& descriptor,
                                               const char* name = nullptr);

    /// Add a QuantizedLstm layer to the network
    /// @param params - The weights and biases for the Quantized LSTM cell
    /// @param name - Optional name for the layer
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddQuantizedLstmLayer(const QuantizedLstmInputParams& params,
                                                     const char* name = nullptr);

    /// Add a QLstm layer to the network
    /// @param descriptor - Parameters for the QLstm operation
    /// @param params - Weights and biases for the layer
    /// @param name - Optional name for the layer
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddQLstmLayer(const QLstmDescriptor& descriptor,
                                             const LstmInputParams& params,
                                             const char* name = nullptr);

    /// Adds a Logical Binary layer to the network.
    /// @param descriptor - Description of the Logical Binary layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddLogicalBinaryLayer(const LogicalBinaryDescriptor& descriptor,
                                                     const char* name = nullptr);

    void Accept(ILayerVisitor& visitor) const;

    void ExecuteStrategy(IStrategy& strategy) const;

protected:
    ~INetwork();

    friend void VisitLayersTopologically(const INetwork* inputNetwork, IStrategy& strategy);
    friend class TestConnectionPreservation;
    friend TensorInfo GetInputTensorInfo(const INetwork* network);
    friend IOptimizedNetworkPtr Optimize(const INetwork& network,
                                         const std::vector<BackendId>& backendPreferences,
                                         const IDeviceSpec& deviceSpec,
                                         const OptimizerOptions& options,
                                         Optional<std::vector<std::string>&> messages);

    INetwork(NetworkOptions networkOptions = {});

    std::unique_ptr<NetworkImpl> pNetworkImpl;
};

namespace experimental
{
class AsyncNetworkImpl;
class WorkingMemHandle;
}

struct BackendSettings;
struct OptimizationResult;
class OptimizedNetworkImpl;
class IOptimizedNetwork
{
public:
    static void Destroy(IOptimizedNetwork* network);

    Status PrintGraph();
    Status SerializeToDot(std::ostream& stream) const;

    profiling::ProfilingGuid GetGuid() const;

    IOptimizedNetwork(std::unique_ptr<Graph> graph);
    IOptimizedNetwork(std::unique_ptr<OptimizedNetworkImpl> impl);
    ~IOptimizedNetwork();

protected:
    friend class LoadedNetwork;

    friend class experimental::AsyncNetworkImpl;
    friend class experimental::WorkingMemHandle;

    friend Graph& GetGraphForTesting(IOptimizedNetwork* optNetPtr);
    friend ModelOptions& GetModelOptionsForTesting(IOptimizedNetwork* optNetPtr);
    friend IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                                         const std::vector<BackendId>& backendPreferences,
                                         const IDeviceSpec& deviceSpec,
                                         const OptimizerOptions& options,
                                         Optional<std::vector<std::string>&> messages);

    IOptimizedNetwork(std::unique_ptr<Graph> graph, const ModelOptions& modelOptions);

    std::unique_ptr<OptimizedNetworkImpl> pOptimizedNetworkImpl;
};

/// Create an optimized version of the network
/// @param network INetwork description of the network to be optimized.
/// @param backendPreferences The choice of the backend ordered by user preferences.
/// @param deviceSpec DeviceSpec object as queried from the runtime. See IRuntime::GetDeviceSpec()
/// @param messages If there are failures or warnings a string describing same will be added to the vector
/// @param options OptimizerOptions object with optimizer configuration options
/// @return An IOptimizedNetworkPtr interface to the optimized network, throws an exception derived from
/// armnn::Exception if process fails.

IOptimizedNetworkPtr Optimize(const INetwork& network,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options = OptimizerOptions(),
                              Optional<std::vector<std::string>&> messages = EmptyOptional());
} //namespace armnn
