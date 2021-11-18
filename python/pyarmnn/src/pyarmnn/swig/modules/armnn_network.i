//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
%{
#include "armnn/INetwork.hpp"
#include "armnn/BackendId.hpp"
#include "armnn/Types.hpp"
#include "armnn/Optional.hpp"
#include <fstream>
%}

%include <typemaps/network_optimize.i>
%include <typemaps/model_options.i>

namespace std {
    %template() std::vector<armnn::BackendOptions>;
}

namespace armnn
{
%feature("docstring",
"
Struct for holding options relating to the Arm NN optimizer. See `Optimize`.

Contains:
    m_debug (bool): Add debug data for easier troubleshooting.
    m_ReduceFp32ToBf16 (bool): Reduces Fp32 network to BFloat16 (Bf16) for faster processing. Layers
                               that can not be reduced will be left in Fp32.
    m_ReduceFp32ToFp16 (bool): Reduces Fp32 network to Fp16 for faster processing. Layers
                               that can not be reduced will be left in Fp32.
    m_ImportEnabled (bool):    Enable memory import.
    m_shapeInferenceMethod:    The ShapeInferenceMethod modifies how the output shapes are treated.
                               When ValidateOnly is selected, the output shapes are inferred from the input parameters
                               of the layer and any mismatch is reported.
                               When InferAndValidate is selected 2 actions are performed: (1)infer output shape from
                               inputs and (2)validate the shapes as in ValidateOnly. This option has been added to work
                               with tensors which rank or dimension sizes are not specified explicitly, however this
                               information can be calculated from the inputs.
    m_ModelOptions:            List of backends optimisation options.

") OptimizerOptions;

%model_options_typemap;
struct OptimizerOptions
{
    OptimizerOptions();

    OptimizerOptions(bool reduceFp32ToFp16,
                     bool debug,
                     bool reduceFp32ToBf16 = false,
                     ShapeInferenceMethod shapeInferenceMethod = armnn::ShapeInferenceMethod::ValidateOnly,
                     bool importEnabled = false,
                     std::vector<armnn::BackendOptions> modelOptions = {});

    bool m_ReduceFp32ToBf16;
    bool m_ReduceFp32ToFp16;
    bool m_Debug;
    ShapeInferenceMethod m_shapeInferenceMethod;
    bool m_ImportEnabled;
    std::vector<armnn::BackendOptions> m_ModelOptions;
};
%model_options_clear;

%feature("docstring",
"
An input connection slot for a layer. Slot lifecycle is managed by the layer.

The input slot can be connected to an output slot of the preceding layer in the graph.
Only one connection to the input slot is allowed.

") IInputSlot;
%nodefaultctor IInputSlot;
%nodefaultdtor IInputSlot;
class IInputSlot
{
public:
    %feature("docstring",
    "
    Returns output slot of a preceding layer that is connected to the given input slot.

    Returns:
        IOutputSlot: Borrowed reference to an output connection slot for a preceding layer.

    ") GetConnection;

    armnn::IOutputSlot* GetConnection();
};

%feature("docstring",
"
An output connection slot for a layer. Slot lifecycle is managed by the layer.

The output slot may be connected to 1 or more input slots of subsequent layers in the graph.
") IOutputSlot;
%nodefaultctor IOutputSlot;
%nodefaultdtor IOutputSlot;
class IOutputSlot
{
public:

    %feature("docstring",
    "
    Returns the total number of connected input slots.

    The same result could be obtained by calling `len()`:

    >>> output_slot = ...
    >>> size = len(output_slot)
    >>> assert size == output_slot.GetNumConnections()

    Returns:
        int: Number of connected input slots.
    ") GetNumConnections;
    unsigned int GetNumConnections();


    %feature("docstring",
    "
    Retrieves connected input slot by index.

    The same result could be obtained by using square brackets:

    >>> output_slot = ...
    >>> connected_input_slot = output_slot[0]

    Args:
       index (int): Slot index.

    Returns:
        IInputSlot: Borrowed reference to connected input slot with given index.

    Raises:
        RuntimeError: If index out of bounds.
    ") GetConnection;
    armnn::IInputSlot* GetConnection(unsigned int index);

    %feature("docstring",
    "
    Sets tensor info for output slot.
    Operation does not change TensorInfo ownership.
    Args:
        tensorInfo (TensorInfo): Output tensor info.

    ") SetTensorInfo;
    void SetTensorInfo(const armnn::TensorInfo& tensorInfo);

    %feature("docstring",
    "
    Gets tensor info for output slot.

    Args:
        tensorInfo (TensorInfo): Output tensor info.

    ") GetTensorInfo;
    const armnn::TensorInfo& GetTensorInfo();

    %feature("docstring",
    "
    Checks if tensor info was set previously.

    Returns:
        bool: True if output tensor info was set, False - otherwise.

    ") IsTensorInfoSet;
    bool IsTensorInfoSet();

    %feature("docstring",
    "
    Connects this output slot with given input slot.
    Input slot is updated with this output connection.

    Args:
        destination (IInputSlot): Output tensor info.

    Returns:
        int: Total number of connections.

    Raises:
        RuntimeError: If input slot was already connected.

    ") Connect;
    int Connect(IInputSlot& destination);

    %feature("docstring",
    "
    Disconnects this output slot from given input slot.

    Args:
        slot (IInputSlot): Input slot to disconnect from.

    ") Disconnect;
    void Disconnect(IInputSlot& slot);

    %feature("docstring",
    "
    Calculates the index of this slot for the layer.

    Returns:
        int: Slot index.

    ") CalculateIndexOnOwner;
    unsigned int CalculateIndexOnOwner();

    %feature("docstring",
    "
    Returns the index of the layer. Same value as `IConnectableLayer.GetGuid`.

    Returns:
        int: Layer id.

    ") GetOwningLayerGuid;
    unsigned int GetOwningLayerGuid();

};

%extend IOutputSlot {

    armnn::IInputSlot* __getitem__(unsigned int index) {
        return $self->GetConnection(index);
    }

    unsigned int __len__() const {
        return $self->GetNumConnections();
    }

}

%feature("docstring",
"
Interface for a layer that is connectable to other layers via `IInputSlot` and `IOutputSlot`.
The object implementing this interface is returned by `INetwork` when calling `add*Layer` methods.

") IConnectableLayer;
%nodefaultctor IConnectableLayer;
%nodefaultdtor IConnectableLayer;
class IConnectableLayer
{
public:
    %feature("docstring",
    "
    Returns the name of the layer. Name attribute is optional for a layer, thus
    `None` value could be returned.

    Returns:
        str: Layer name or `None`.

    ") GetName;
    const char* GetName();

    %feature("docstring",
    "
    Gets the number of input slots for the layer.

    Returns:
        int: Number of input slots.

    ") GetNumInputSlots;
    unsigned int GetNumInputSlots();

    %feature("docstring",
    "
    Gets the number of output slots for the layer.

    Returns:
        int: Number of output slots.

    ") GetNumOutputSlots;
    unsigned int GetNumOutputSlots();

    %feature("docstring",
    "
    Gets the input slot by index.

    Args:
        index (int): Slot index.

    Returns:
        IInputSlot: Borrowed reference to input slot.

    ") GetInputSlot;
    armnn::IInputSlot& GetInputSlot(unsigned int index);

    %feature("docstring",
    "
    Gets the output slot by index.

    Args:
        index (int): Slot index.

    Returns:
        IOutputSlot: Borrowed reference to output slot.

    ") GetOutputSlot;
    armnn::IOutputSlot& GetOutputSlot(unsigned int index);


    %feature("docstring",
    "
    Gets the unique layer id (within one process).
    Guid is generated and assigned automatically when the layer is created.

    Returns:
        int: The unique layer id.

    ") GetGuid;
    unsigned int GetGuid();
};

%feature("docstring",
    "
    Interface for a network object. Network objects contain the whole computation graph, made up of different layers connected together.

    INetwork objects can be constructed manually or obtained by using parsers. INetwork objects are used to create optimized networks, see `Optimize`.

    ") INetwork;
%nodefaultctor INetwork;
%nodefaultdtor INetwork;
class INetwork
{
public:

    %feature("docstring",
        "
        Adds an input layer to the network. Input layers are placed at the start of a network and used for feeding input data during inference.

        Args:
            id (int): User generated id to uniquely identify a particular input. The same id needs to be specified
                      when passing the inputs to the IRuntime::EnqueueWorkload() function.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddInputLayer;
    armnn::IConnectableLayer* AddInputLayer(int id, const char* name = nullptr);

    %feature("docstring",
        "
        Adds an addition layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddAdditionLayer;
    armnn::IConnectableLayer* AddAdditionLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds an output layer to the network. Output layer is the final layer in your network.

    Args:
        id (int): User generated id to uniquely identify a particular input. The same id needs to be specified
                  when passing the inputs to `IRuntime.EnqueueWorkload()`.
        name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddOutputLayer;
    armnn::IConnectableLayer* AddOutputLayer(int id, const char* name = nullptr);


    %feature("docstring",
        "
        Adds an Activation layer to the network. Type of activation is decided by activationDescriptor.

        Args:
            activationDescriptor (ActivationDescriptor): ActivationDescriptor to configure the activation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddActivationLayer;
    armnn::IConnectableLayer* AddActivationLayer(const ActivationDescriptor& activationDescriptor,
        const char* name = nullptr);


    %feature("docstring",
        "
        Adds an ArgMinMax layer to the network.

        Args:
            desc (ArgMinMaxDescriptor): Parameters for the ArgMinMax layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddArgMinMaxLayer;
    armnn::IConnectableLayer* AddArgMinMaxLayer(const armnn::ArgMinMaxDescriptor& desc,
                                                 const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Batch Normalization layer to the network.

        Args:
            mean (ConstTensor): Pre-calculated mean for each channel.
            variance (ConstTensor): Pre-calculated variance for each channel.
            beta (ConstTensor): Per-channel additive factor.
            gamma (ConstTensor): Per-channel multiplicative factor.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddBatchNormalizationLayer;
    armnn::IConnectableLayer* AddBatchNormalizationLayer(const armnn::BatchNormalizationDescriptor& desc,
        const armnn::ConstTensor& mean,
        const armnn::ConstTensor& variance,
        const armnn::ConstTensor& beta,
        const armnn::ConstTensor& gamma,
        const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Batch To Space ND layer to the network.

        Args:
            batchToSpaceNdDescriptor (BatchToSpaceNdDescriptor): Configuration parameters for the layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddBatchToSpaceNdLayer;
    armnn::IConnectableLayer* AddBatchToSpaceNdLayer(const armnn::BatchToSpaceNdDescriptor& batchToSpaceNdDescriptor,
                                                     const char* name = nullptr);

    %feature("docstring",
         "
        Adds a ChannelShuffle layer to the network.

        Args:
            channelShuffleDescriptor (ChannelShuffleDescriptor): Configuration parameters for the layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddChannelShuffleLayer;
    armnn::IConnectableLayer* AddChannelShuffleLayer(const armnn::ChannelShuffleDescriptor& channelShuffleDescriptor,
                                                     const char* name = nullptr);



    %feature("docstring",
        "
        Adds a Cast layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddCastLayer;
    armnn::IConnectableLayer* AddCastLayer(const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Comparison layer to the network.

        Args:
            comparisonDescriptor (ComparisonDescriptor): Configuration parameters for the layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddComparisonLayer;
    armnn::IConnectableLayer* AddComparisonLayer(const armnn::ComparisonDescriptor& comparisonDescriptor,
                                                 const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Concatenation layer to the network.

        Args:
            concatDescriptor (ConcatDescriptor): Parameters to configure the Concatenation layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddConcatLayer;
    armnn::IConnectableLayer* AddConcatLayer(const armnn::ConcatDescriptor& concatDescriptor,
                                             const char* name = nullptr);


    %feature("docstring",
        "
        Adds a layer with no inputs and a single output, which always corresponds to the passed in constant tensor.

        Args:
            input (ConstTensor): Tensor to be provided as the only output of the layer. The layer will maintain
                    its own copy of the tensor data, meaning the memory referenced by input can
                    be freed or reused after this function is called.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddConstantLayer;
    armnn::IConnectableLayer* AddConstantLayer(const armnn::ConstTensor& input,
                                               const char* name = nullptr);


    %feature("docstring",
             "
    Adds a 3D Convolution layer to the network.

            Args:
    convolution3dDescriptor (Convolution3dDescriptor): Description of the 3D convolution layer.
            name (str): Optional name for the layer.

            Returns:
    IConnectableLayer: Interface for configuring the layer.
    ") AddConvolution3dLayer;

    armnn::IConnectableLayer* AddConvolution3dLayer(const armnn::Convolution3dDescriptor& convolution3dDescriptor,
    const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Depth To Space layer to the network.

        Args:
            depthToSpaceDescriptor (DepthToSpaceDescriptor): Parameters for the depth to space operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddDepthToSpaceLayer;
    armnn::IConnectableLayer* AddDepthToSpaceLayer(const armnn::DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                                   const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Dequantize layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddDequantizeLayer;
    armnn::IConnectableLayer* AddDequantizeLayer(const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Detection PostProcess layer to the network. Detection PostProcess is a custom layer for SSD MobilenetV1.

        Args:
            descriptor (DetectionPostProcessDescriptor): Description of the Detection PostProcess layer.
            anchors (ConstTensor): Tensor for anchors.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddDetectionPostProcessLayer;
    armnn::IConnectableLayer* AddDetectionPostProcessLayer(
        const armnn::DetectionPostProcessDescriptor& descriptor,
        const armnn::ConstTensor& anchors,
        const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Division layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddDivisionLayer;
    armnn::IConnectableLayer* AddDivisionLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds an Elementwise Unary layer to the network. Type of unary operation to use is decided by elementwiseUnaryDescriptor. Unary operations supported are (Abs, Exp, Neg, Rsqrt, Sqrt)

        Args:
            elementwiseUnaryDescriptor (ElementwiseUnaryDescriptor): ElementwiseUnaryDescriptor to configure the choice of unary operation added to the network.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddElementwiseUnaryLayer;
    armnn::IConnectableLayer* AddElementwiseUnaryLayer(const ElementwiseUnaryDescriptor& elementwiseUnaryDescriptor,
                                                       const char* name = nullptr);

    %feature("docstring",
        "
        Add a Fill layer to the network.

        Args:
            FillDescriptor (FillDescriptor): Descriptor for the fill operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddFillLayer;
    armnn::IConnectableLayer* AddFillLayer(const FillDescriptor& fillDescriptor,
                                           const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Floor layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddFloorLayer;
    armnn::IConnectableLayer* AddFloorLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Add Gather layer to the network.

        Args:
            descriptor (GatherDescriptor): Descriptor for the gather operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddGatherLayer;
    armnn::IConnectableLayer* AddGatherLayer(const GatherDescriptor& descriptor,
                                             const char* name = nullptr);

    %feature("docstring",
        "
        Adds an Instance Normalization layer to the network.

        Args:
            desc (InstanceNormalizationDescriptor): Parameters for the instance normalization operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddInstanceNormalizationLayer;
    armnn::IConnectableLayer* AddInstanceNormalizationLayer(const armnn::InstanceNormalizationDescriptor& desc,
                                                            const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Log Softmax layer to the network.

        Args:
            desc (SoftmaxDescriptor): parameters to configure the log softmax.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddLogSoftmaxLayer;
    armnn::IConnectableLayer* AddLogSoftmaxLayer(const armnn::LogSoftmaxDescriptor& logSoftmaxDescriptor,
                                                  const char* name = nullptr);

    %feature("docstring",
        "
        Adds an L2 Normalization layer to the network.
        Normalization is performed along dimension 1, but requires a 4d input.

        Args:
            desc (L2NormalizationDescriptor): Parameters for the L2 normalization operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddL2NormalizationLayer;
    armnn::IConnectableLayer* AddL2NormalizationLayer(const armnn::L2NormalizationDescriptor& desc,
                                                       const char* name = nullptr);

    %feature("docstring",
        "
        Add a Long Short-Term Memory layer to the network.

        Args:
            descriptor (LstmDescriptor): Parameters for the Lstm operation.
            params (LstmInputParams): Weights and biases for the LSTM cell.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddLstmLayer;
    armnn::IConnectableLayer* AddLstmLayer(const armnn::LstmDescriptor& descriptor,
                                            const armnn::LstmInputParams& params,
                                            const char* name = nullptr);

    %feature("docstring",
        "
        Add a Maximum layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddMaximumLayer;
    armnn::IConnectableLayer* AddMaximumLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Mean layer to the network.

        Args:
            meanDescriptor (meanDescriptor): Parameters for the mean operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddMeanLayer;
    armnn::IConnectableLayer* AddMeanLayer(const armnn::MeanDescriptor& meanDescriptor, const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Merge layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddMergeLayer;
    armnn::IConnectableLayer* AddMergeLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Minimum layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddMinimumLayer;
    armnn::IConnectableLayer* AddMinimumLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Multiplication layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddMultiplicationLayer;
    armnn::IConnectableLayer* AddMultiplicationLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Normalization layer to the network.

        Args:
            normalizationDescriptor (NormalizationDescriptor): Parameters to configure the normalization.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddNormalizationLayer;
    armnn::IConnectableLayer* AddNormalizationLayer(const armnn::NormalizationDescriptor& normalizationDescriptor,
        const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Pad layer to the network.

        Args:
            padDescriptor (PadDescriptor): Padding configuration for the layer. See `PadDescriptor` for more details.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddPadLayer;
    armnn::IConnectableLayer* AddPadLayer(const armnn::PadDescriptor& padDescriptor,
                                           const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Permute layer to the network.

        Args:
            permuteDescriptor (PermuteDescriptor): Configuration of the permutation layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddPermuteLayer;
    armnn::IConnectableLayer* AddPermuteLayer(const armnn::PermuteDescriptor& permuteDescriptor,
                                               const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Pooling layer to the network. Type of pooling is decided by the configuration.

        Args:
            pooling2dDescriptor (Pooling2dDescriptor): Configuration for the pooling layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddPooling2dLayer;
    armnn::IConnectableLayer* AddPooling2dLayer(const armnn::Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr);

    %feature("docstring",
        "
        Adds a PReLU layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddPreluLayer;
    armnn::IConnectableLayer* AddPreluLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Quantize layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddQuantizeLayer;
    armnn::IConnectableLayer* AddQuantizeLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Quantized Long Short-Term Memory layer to the network.

        Args:
            params (`QuantizedLstmInputParams`): The weights and biases for the Quantized LSTM cell.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddQuantizedLstmLayer;
    armnn::IConnectableLayer* AddQuantizedLstmLayer(const armnn::QuantizedLstmInputParams& params,
                                                     const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Rank layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddRankLayer;
    armnn::IConnectableLayer* AddRankLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Reduce layer to the network.

        Args:
            reduceDescriptor (ReduceDescriptor): Parameters for the reduce operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddReduceLayer;
    armnn::IConnectableLayer* AddReduceLayer(const armnn::ReduceDescriptor& reduceDescriptor,
                                             const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Reshape layer to the network.

        Args:
            reshapeDescriptor (ReshapeDescriptor): Parameters for the reshape operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddReshapeLayer;
    armnn::IConnectableLayer* AddReshapeLayer(const armnn::ReshapeDescriptor& reshapeDescriptor,
                                               const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Resize layer to the network.

        Args:
            resizeDescriptor (ResizeDescriptor): Configuration for the resize layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddResizeLayer;
    armnn::IConnectableLayer* AddResizeLayer(const armnn::ResizeDescriptor& resizeDescriptor,
                                              const char* name = nullptr);


    %feature("docstring",
        "
        Adds a Shape layer to the network.

        Args:
            name(str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer
        ") AddShapeLayer;
    armnn::IConnectableLayer* AddShapeLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Slice layer to the network.

        Args:
            sliceDescriptor (SliceDescriptor): Descriptor to configure the slice operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSliceLayer;
    armnn::IConnectableLayer* AddSliceLayer(const armnn::SliceDescriptor& sliceDescriptor,
                                            const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Softmax layer to the network.

        If the data type is `DataType_QuantisedAsymm8`, then the output quantization parameters
        must have a scale of 1/256 and an offset of 0.

        Args:
            softmaxDescriptor (SoftmaxDescriptor): Configuration for the softmax layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSoftmaxLayer;
    armnn::IConnectableLayer* AddSoftmaxLayer(const armnn::SoftmaxDescriptor& softmaxDescriptor,
        const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Space To Batch layer to the network.

        Args:
            spaceToBatchNdDescriptor (SpaceToBatchNdDescriptor): Configuration for the space to batch layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSpaceToBatchNdLayer;
    armnn::IConnectableLayer* AddSpaceToBatchNdLayer(const armnn::SpaceToBatchNdDescriptor& spaceToBatchNdDescriptor,
                                                      const char* name = nullptr);

    %feature("docstring",
        "
        Adds a space to depth layer to the network.

        Args:
            spaceToDepthDescriptor (SpaceToDepthDescriptor): Parameters for the space to depth operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSpaceToDepthLayer;
    armnn::IConnectableLayer* AddSpaceToDepthLayer(const armnn::SpaceToDepthDescriptor& spaceToDepthDescriptor,
                                                    const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Splitter layer to the network.

        Args:
            splitterDescriptor (SplitterDescriptor): Parameters to configure the splitter layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSplitterLayer;
    armnn::IConnectableLayer* AddSplitterLayer(const armnn::SplitterDescriptor& splitterDescriptor,
                                                const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Stack layer to the network.

        Args:
            descriptor (StackDescriptor):  Descriptor to configure the stack layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddStackLayer;
    armnn::IConnectableLayer* AddStackLayer(const armnn::StackDescriptor& descriptor,
                                             const char* name = nullptr);

    %feature("docstring",
        "
        Adds a StandIn layer to the network.

        Args:
            descriptor (StandInDescriptor): Parameters to configure the standIn layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddStandInLayer;
    armnn::IConnectableLayer* AddStandInLayer(const armnn::StandInDescriptor& descriptor,
                                              const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Strided Slice layer to the network.

        Args:
            stridedSliceDescriptor (StridedSliceDescriptor): Parameters for the strided slice operation.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddStridedSliceLayer;
    armnn::IConnectableLayer* AddStridedSliceLayer(const armnn::StridedSliceDescriptor& stridedSliceDescriptor,
                                                   const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Subtraction layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSubtractionLayer;
    armnn::IConnectableLayer* AddSubtractionLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Switch layer to the network.

        Args:
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddSwitchLayer;
    armnn::IConnectableLayer* AddSwitchLayer(const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Fully Connected layer to the network. Also known as a Linear or Dense layer.

        Args:
            fullyConnectedDescriptor (FullyConnectedDescriptor): Description of the fully connected layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddFullyConnectedLayer;
    armnn::IConnectableLayer* AddFullyConnectedLayer(const armnn::FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                     const char* name = nullptr);

    %feature("docstring",
        "
        Adds a LogicalBinary layer to the network.

        Args:
            logicalBinaryDescriptor (LogicalBinaryDescriptor): Description of the LogicalBinary layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddLogicalBinaryLayer;
    armnn::IConnectableLayer* AddLogicalBinaryLayer(const armnn::LogicalBinaryDescriptor& logicalBinaryDescriptor,
                                                    const char* name = nullptr);

    %feature("docstring",
        "
        Adds a Transpose layer to the network.

        Args:
            transposeDescriptor (TransposeDescriptor): Description of the transpose layer.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddTransposeLayer;
    armnn::IConnectableLayer* AddTransposeLayer(const armnn::TransposeDescriptor& transposeDescriptor,
                                                const char* name = nullptr);
};

%extend INetwork {

    INetwork() {
        return armnn::INetwork::CreateRaw();
    }

    ~INetwork() {
        armnn::INetwork::Destroy($self);
    }

    %feature("docstring",
        "
        Adds a Fully Connected layer to the network with input weights and optional bias.
        Also known as a Linear or Dense layer.

        Args:
            fullyConnectedDescriptor (FullyConnectedDescriptor): Description of the fully connected layer.
            weights (ConstTensor): Tensor for the weights data.
            biases (ConstTensor): Optional tensor for the bias data.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
    ") AddFullyConnectedLayer;
    armnn::IConnectableLayer* AddFullyConnectedLayer(const armnn::FullyConnectedDescriptor& fullyConnectedDescriptor,
                                                     const armnn::ConstTensor& weights,
                                                     armnn::ConstTensor* biases = nullptr,
                                                     const char* name = nullptr) {
        ARMNN_NO_DEPRECATE_WARN_BEGIN
        if (biases) {
            return $self->AddFullyConnectedLayer(fullyConnectedDescriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(*biases), name);
        } else {
            return $self->AddFullyConnectedLayer(fullyConnectedDescriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(), name);
        }
        ARMNN_NO_DEPRECATE_WARN_END
    }

    %feature("docstring",
        "
        Adds a 2D Transpose Convolution layer to the network.

        Args:
            descriptor (TransposeConvolution2dDescriptor): Descriptor containing all parameters to configure this layer.
            weights (ConstTensor): Tensor for the weights data.
            biases (ConstTensor): Optional tensor for the bias data.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddTransposeConvolution2dLayer;
    armnn::IConnectableLayer* AddTransposeConvolution2dLayer(const armnn::TransposeConvolution2dDescriptor& descriptor,
                                                             const armnn::ConstTensor& weights,
                                                             armnn::ConstTensor* biases = nullptr,
                                                             const char* name = nullptr) {

        if (biases) {
            return $self->AddTransposeConvolution2dLayer(descriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(*biases), name);
        } else {
            return $self->AddTransposeConvolution2dLayer(descriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(), name);
        }
    }


    %feature("docstring",
        "
        Adds a 2D Convolution layer to the network.

        Args:
            convolution2dDescriptor (Convolution2dDescriptor): Description of the 2D convolution layer.
            weights (ConstTensor): Tensor for the weights data.
            biases (ConstTensor): Optional tensor for the bias data. If specified, must match the output tensor shape.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddConvolution2dLayer;
    armnn::IConnectableLayer* AddConvolution2dLayer(const armnn::Convolution2dDescriptor& convolution2dDescriptor,
                                                    const armnn::ConstTensor& weights,
                                                    armnn::ConstTensor* biases = nullptr,
                                                    const char* name = nullptr) {

        if (biases) {
            return $self->AddConvolution2dLayer(convolution2dDescriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(*biases), name);
        } else {
            return $self->AddConvolution2dLayer(convolution2dDescriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(), name);
        }
    }


    %feature("docstring",
        "
        Adds a 2D Depthwise Convolution layer to the network.

        Args:
            convolution2dDescriptor (DepthwiseConvolution2dDescriptor): Description of the 2D depthwise convolution layer.
            weights (ConstTensor): Tensor for the weights. Expected format: [channelMultiplier, inputChannels, height, width].
            biases (ConstTensor): Optional tensor for the bias data. If specified, must match the output tensor shape.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") AddDepthwiseConvolution2dLayer;

    armnn::IConnectableLayer* AddDepthwiseConvolution2dLayer(
        const armnn::DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
        const armnn::ConstTensor& weights,
        const armnn::ConstTensor* biases = nullptr,
        const char* name = nullptr) {

        if (biases) {
            return $self->AddDepthwiseConvolution2dLayer(convolution2dDescriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(*biases), name);
        } else {
            return $self->AddDepthwiseConvolution2dLayer(convolution2dDescriptor, weights,
                                                 armnn::Optional<armnn::ConstTensor>(), name);
        }
    }
}

%feature("docstring",
        "
        Interface class for an optimzied network object. Optimized networks are obtained after running `Optimize` on
        an `INetwork` object.
        Optimized networks are passed to `EnqueueWorkload`.

        Args:
            convolution2dDescriptor (DepthwiseConvolution2dDescriptor): Description of the 2D depthwise convolution layer.
            weights (ConstTensor): Tensor for the weights. Expected format: [channelMultiplier, inputChannels, height, width].
            biases (ConstTensor): Optional tensor for the bias data. If specified, must match the output tensor shape.
            name (str): Optional name for the layer.

        Returns:
            IConnectableLayer: Interface for configuring the layer.
        ") IOptimizedNetwork;
%nodefaultctor IOptimizedNetwork;
%nodefaultdtor IOptimizedNetwork;
class IOptimizedNetwork
{
};

%extend IOptimizedNetwork {

    ~IOptimizedNetwork() {
        armnn::IOptimizedNetwork::Destroy($self);
    }

    %feature("docstring",
        "
        Saves optimized network graph as dot file.

        Args:
            fileName (str): File path to save to.
        Raises:
            RuntimeError: If serialization failure.
        ") SerializeToDot;

    void SerializeToDot(const std::string& fileName) {
        std::ofstream dot;
        dot.open(fileName);
        if(!dot.is_open())
        {
            throw armnn::Exception("Failed to open dot file");
        } else {
            armnn::Status status = $self->SerializeToDot(dot);
            dot.close();
            if(status == armnn::Status::Failure)
            {
                throw armnn::Exception("Failed to serialize to dot");
            }
        }
    };
}
}

%{
    std::pair<armnn::IOptimizedNetwork*, std::vector<std::string>> Optimize(const armnn::INetwork* network,
                                       const std::vector<armnn::BackendId>& backendPreferences,
                                       const armnn::IDeviceSpec& deviceSpec,
                                       const armnn::OptimizerOptions& options = armnn::OptimizerOptions())
    {
        std::vector<std::string> errorMessages;
        armnn::IOptimizedNetwork* optimizedNetwork = armnn::Optimize(*network, backendPreferences, deviceSpec,
            options, armnn::Optional<std::vector<std::string> &>(errorMessages)).release();

        if(!optimizedNetwork)
        {
            std::string errorString;

            for (auto error : errorMessages) {
                errorString.append(error);
            }

            throw armnn::Exception(errorString);
        }

        return std::make_pair(optimizedNetwork, errorMessages);
    };
%}

%feature("docstring",
    "
    Create an optimized version of the given network. Should be called before loading a network into the runtime.

    Examples:
        Optimize a loaded network ready for inference.
        >>> parser = ann.ITfLiteParser()
        >>> network = parser.CreateNetworkFromBinaryFile('./model.tflite')
        >>>
        >>> preferredBackends = [ann.BackendId('CpuAcc'), ann.BackendId('CpuRef')]
        >>> opt_network, messages = ann.Optimize(network, preferredBackends, runtime.GetDeviceSpec(), ann.OptimizerOptions())

    Args:
        network (INetwork): INetwork description of the network to be optimized.
        backendPreferences (list): The choice of the backend ordered by user preferences. See `BackendId`.
        deviceSpec (IDeviceSpec): DeviceSpec object as queried from the runtime. See `IRuntime.GetDeviceSpec`.
        options (OptimizerOptions): Object with optimizer configuration options.

    Returns:
        tuple: (`IOptimizedNetwork`, a tuple of failures or warnings).

    Raises:
        RuntimeError: If process fails.
    ") Optimize;

%optimize_typemap_out;
std::pair<armnn::IOptimizedNetwork*, std::vector<std::string>> Optimize(const armnn::INetwork* network,
                                   const std::vector<armnn::BackendId>& backendPreferences,
                                   const armnn::IDeviceSpec& deviceSpec,
                                   const armnn::OptimizerOptions& options = OptimizerOptions());
%clear_optimize_typemap_out;
