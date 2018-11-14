//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/NetworkFwd.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Optional.hpp>

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

protected:
    /// Not user deletable.
    ~IOutputSlot() {}
};

/// @brief Interface for a layer that is connectable to other layers via InputSlots and OutputSlots.
class IConnectableLayer
{
public:
    virtual const char* GetName() const = 0;

    virtual unsigned int GetNumInputSlots() const = 0;
    virtual unsigned int GetNumOutputSlots() const = 0;

    virtual const IInputSlot& GetInputSlot(unsigned int index) const = 0;
    virtual IInputSlot& GetInputSlot(unsigned int index) = 0;

    virtual const IOutputSlot& GetOutputSlot(unsigned int index) const = 0;
    virtual IOutputSlot& GetOutputSlot(unsigned int index) = 0;

    virtual std::vector<TensorShape> InferOutputShapes(const std::vector<TensorShape>& inputShapes) const = 0;

    virtual LayerGuid GetGuid() const = 0;
protected:
      /// Objects are not deletable via the handle
    ~IConnectableLayer() {}
};

using INetworkPtr = std::unique_ptr<INetwork, void(*)(INetwork* network)>;

/// Main network class which provides the interface for building up a neural network.
/// This object is subsequently required by the IRuntime::Load() method.
class INetwork
{
public:
    static INetwork* CreateRaw();
    static INetworkPtr Create();
    static void Destroy(INetwork* network);

    virtual Status PrintGraph() = 0;

    /// Adds an input layer to the network.
    /// @param id - User generated id to uniquely identify a particular input. The same id needs to be specified.
    /// when passing the inputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddInputLayer(LayerBindingId id, const char* name = nullptr) = 0;

    /// Adds a 2D convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D convolution layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - (Optional) Tensor for the bias data. Must match the output tensor shape.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) = 0;

    virtual IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) = 0;

    /// Adds a 2D depthwise convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D depthwise convolution layer.
    /// @param weights - Tensor for the weights data. Expected format: [1, outputChannels, height, width].
    /// @param biases (Optional) - Tensor for the bias data. Must match the output tensor shape.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) = 0;

    virtual IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) = 0;

    /// Adds a fully connected layer to the network.
    /// @param fullyConnectedDescriptor - Description of the fully connected layer.
    /// @param weights - Tensor for the weights data.
    /// @param biases - (Optional) Tensor for the bias data.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const char* name = nullptr) = 0;

    virtual IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
        const ConstTensor& weights,
        const ConstTensor& biases,
        const char* name = nullptr) = 0;

    /// Adds a permute layer to the network.
    /// @param permuteDescriptor - PermuteDescriptor to configure the permute.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddPermuteLayer(const PermuteDescriptor& permuteDescriptor,
                                               const char* name = nullptr) = 0;

    /// Adds a batch to space ND layer to the network.
    /// @param batchToSpaceNdDescriptor - Description of the layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddBatchToSpaceNdLayer(const BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                                      const char* name = nullptr) = 0;

    /// Adds a pooling layer to the network.
    /// @param pooling2dDescriptor - Pooling2dDescriptor to configure the pooling.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr) = 0;

    /// Adds an activation layer to the network.
    /// @param activationDescriptor - ActivationDescriptor to configure the activation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr) = 0;

    /// Adds a normalization layer to the network.
    /// @param normalizationDescriptor - NormalizationDescriptor to configure the normalization.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddNormalizationLayer(const NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr) = 0;

    /// Adds a softmax layer to the network.
    /// @param softmaxDescriptor - SoftmaxDescriptor to configure the softmax.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddSoftmaxLayer(const SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr) = 0;

    /// Adds a splitter layer to the network.
    /// @param splitterDescriptor - WindowsDescriptor to configure the splitting process.
    ///                             Number of Views must be equal to the number of outputs,
    ///                             and their order must match - e.g. first view corresponds to
    ///                             the first output, second view to the second output, etc....
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddSplitterLayer(const ViewsDescriptor& splitterDescriptor
        , const char* name = nullptr) = 0;

    /// Adds a merger layer to the network.
    /// @param mergerDescriptor - WindowsDescriptor to configure the merging process. Number of Views must be equal to
    ///                           the number of inputs, and their order must match - e.g. first view corresponds to
    ///                           the first input, second view to the second input, etc....
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddMergerLayer(const OriginsDescriptor& mergerDescriptor,
        const char* name = nullptr) = 0;

    /// Adds an addition layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddAdditionLayer(const char* name = nullptr) = 0;

    /// Adds a multiplication layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr) = 0;

    /// Adds a batch normalization layer to the network.
    /// @param mean - Pre-calculated mean for each channel.
    /// @param variance - Pre-calculated variance for each channel.
    /// @param beta - Per-channel additive factor.
    /// @param gamma - Per-channel multiplicative factor.
    /// @return - Interface for configuring the layer.
    /// @param name - Optional name for the layer.
    virtual IConnectableLayer* AddBatchNormalizationLayer(const BatchNormalizationDescriptor& desc,
        const ConstTensor& mean,
        const ConstTensor& variance,
        const ConstTensor& beta,
        const ConstTensor& gamma,
        const char* name = nullptr) = 0;

    /// Adds a resize bilinear layer to the network.
    /// @param resizeDesc - Parameters for the resize operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddResizeBilinearLayer(const ResizeBilinearDescriptor& resizeDesc,
                                                      const char* name = nullptr) = 0;

    /// Adds an L2 normalization layer to the network.
    /// Normalization is performed along dimension 1, but requires a 4d input.
    /// @param desc - Parameters for the L2 normalization operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddL2NormalizationLayer(const L2NormalizationDescriptor& desc,
                                                       const char* name = nullptr) = 0;

    /// Adds a layer with no inputs and a single output, which always corresponds to
    /// the passed in constant tensor.
    /// @param input - Tensor to be provided as the only output of the layer. The layer will maintain
    ///                its own copy of the tensor data, meaning the memory referenced by @a input can
    ///                be freed or reused after this function is called.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddConstantLayer(const ConstTensor& input,
        const char* name = nullptr) = 0;

    /// Adds a reshape layer to the network.
    /// @param reshapeDescriptor - Parameters for the reshape operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddReshapeLayer(const ReshapeDescriptor& reshapeDescriptor,
                                               const char* name = nullptr) = 0;

    /// Adds a space to batch layer to the network.
    /// @param spaceToBatchNdDescriptor - Parameters for the space to batch operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddSpaceToBatchNdLayer(const SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                      const char* name = nullptr) = 0;

    /// Adds a floor layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddFloorLayer(const char* name = nullptr) = 0;

    /// Adds an output layer to the network.
    /// @param id - User generated id to uniquely identify a particular output. The same id needs to be specified
    /// when passing the outputs to the IRuntime::EnqueueWorkload() function.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddOutputLayer(LayerBindingId id, const char* name = nullptr) = 0;

    /// Add a Lstm layer to the network
    /// @param descriptor Parameters for the Lstm operation
    /// @param name Optional name for the layer
    /// @return Interface for configuring the layer.
    virtual IConnectableLayer* AddLstmLayer(const LstmDescriptor& descriptor,
                                            const LstmInputParams& params,
                                            const char* name = nullptr) = 0;

    /// Adds a division layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddDivisionLayer(const char* name = nullptr) = 0;

    /// Adds a subtraction layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddSubtractionLayer(const char* name = nullptr) = 0;

    /// Add a Mean layer to the network.
    /// @param meanDescriptor - Parameters for the mean operation.
    /// @param name - Optional name for the layer.
    /// @ return - Interface for configuring the layer.
    virtual IConnectableLayer* AddMeanLayer(const MeanDescriptor& meanDescriptor, const char* name = nullptr) = 0;

    /// Adds a fully pad layer to the network.
    /// @param paddings - n by 2 tensor, where n is the rank of the input tensor,
    ///                   such that paddings[i,0] indicates the amount of padding to add in front of dimonsion i, and
    ///                   paddings[i,1] indicates the amount of padding to add after the end of dimension i
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddPadLayer(const PadDescriptor& padDescriptor,
                                           const char* name = nullptr) = 0;

    /// Adds a strided slice layer to the network.
    /// @param StridedSliceDescriptor - Parameters for the strided slice operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    virtual IConnectableLayer* AddStridedSliceLayer(const StridedSliceDescriptor& stridedSliceDescriptor,
                                                    const char* name = nullptr) = 0;

protected:
    ~INetwork() {}
};

using IOptimizedNetworkPtr = std::unique_ptr<IOptimizedNetwork, void(*)(IOptimizedNetwork* network)>;

class IOptimizedNetwork
{
public:
    static void Destroy(IOptimizedNetwork* network);

    virtual Status PrintGraph() = 0;
    virtual Status SerializeToDot(std::ostream& stream) const = 0;


protected:
    ~IOptimizedNetwork() {}
};

struct OptimizerOptions
{
    OptimizerOptions() : m_ReduceFp32ToFp16(false) {}

    OptimizerOptions(bool reduceFp32ToFp16)
        : m_ReduceFp32ToFp16(reduceFp32ToFp16)
    {
    }

    // Reduce Fp32 data to Fp16 for faster processing
    bool m_ReduceFp32ToFp16;
};

/// Create an optimized version of the network
/// @param network INetwork description of the network to be optimized.
/// @param backendPreferences The choice of the backend ordered by user preferences.
/// @param deviceSpec DeviceSpec object as queried from the runtime. See IRuntime::GetDeviceSpec()
/// @param errMessages if there are failures or warnings a string describing same will be added to the vector
/// @param options OptimizerOptions object with optimizer configuration options
/// @return An IOptimizedNetworkPtr interface to the optimized network, throws an exception derived from
/// armnn::Exception if process fails.

IOptimizedNetworkPtr Optimize(const INetwork& network,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options = OptimizerOptions(),
                              Optional<std::vector<std::string>&> errMessages = EmptyOptional());
} //namespace armnn
