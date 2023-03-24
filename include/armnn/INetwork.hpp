//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/BackendOptions.hpp>
#include <armnn/Deprecated.hpp>
#include <armnn/DescriptorsFwd.hpp>
#include <armnn/IStrategy.hpp>
#include <armnn/NetworkFwd.hpp>
#include <armnn/Optional.hpp>
#include <armnn/TensorFwd.hpp>
#include <armnn/Logging.hpp>
#include <armnn/backends/TensorHandle.hpp>

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
    virtual const IConnectableLayer& GetOwningIConnectableLayer() const = 0;
    virtual IConnectableLayer& GetOwningIConnectableLayer() = 0;
    virtual unsigned int GetSlotIndex() const = 0;

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
    virtual IInputSlot* GetConnection(unsigned int outputindex) = 0;

    virtual void SetTensorInfo(const TensorInfo& tensorInfo) = 0;
    virtual const TensorInfo& GetTensorInfo() const = 0;
    virtual bool IsTensorInfoSet() const = 0;

    virtual int Connect(IInputSlot& destination) = 0;
    virtual void Disconnect(IInputSlot& slot) = 0;

    virtual unsigned int CalculateIndexOnOwner() const = 0;

    virtual LayerGuid GetOwningLayerGuid() const = 0;

    virtual const IConnectableLayer& GetOwningIConnectableLayer() const = 0;
    virtual IConnectableLayer& GetOwningIConnectableLayer() = 0;

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
    virtual void ExecuteStrategy(IStrategy& strategy) const = 0;

    /// Provide a hint for the optimizer as to which backend to prefer for this layer.
    /// By providing a BackendSelectionHint there is no guarantee the input backend supports that layer.
    /// If IsLayerSupported() returns false with the backend hint, we default to calling IsLayerSupported()
    /// on the BackendPreferences vector. Use SetBackendId() if we can guarantee a backend supports that
    /// layer (IsLayerSupported returns true for a specific backend).
    virtual void BackendSelectionHint(Optional<BackendId> backend) = 0;

    /// Returns the armnn::LayerType of this layer
    virtual LayerType GetType() const = 0;

    /// If the layer has a descriptor return it.
    /// The base descriptor can then be cast to the correct descriptor class.
    /// If the layer has no associated descriptor a struct of type NullDescriptor will be returned.
    /// Note: NullDescriptors can be detected because they return true when
    /// the BaseDescriptor IsNull function is invoked.
    virtual const BaseDescriptor& GetParameters() const = 0;

    /// Set the backend of the IConnectableLayer.
    /// By using SetBackendId() we guarantee that the input backend supports that
    /// layer (IsLayerSupported returns true for a specific backend). If there is
    /// no guarantee the input backend supports that layer use BackendSelectionHint().
    virtual void SetBackendId(const BackendId& id) = 0;

    using ConstantTensors = std::vector<std::reference_wrapper<std::shared_ptr<ConstTensorHandle>>>;

    // Returns ConstantTensors of this Layer if it has any, otherwise returns empty vector.
    virtual ConstantTensors GetConstantTensorsByRef() = 0;

    using ImmutableConstantTensors = std::vector<std::reference_wrapper<const std::shared_ptr<ConstTensorHandle>>>;

    // Returns ConstantTensors of this Layer if it has any, otherwise returns empty vector.
    virtual ImmutableConstantTensors GetConstantTensorsByRef() const = 0;

protected:
      /// Objects are not deletable via the handle
    ~IConnectableLayer() {}
};

struct OptimizerOptions
{
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable OptimizerOptionsOpaque instead.", "24.02")
    OptimizerOptions()
            : m_ReduceFp32ToFp16(false)
            , m_Debug(false)
            , m_DebugToFile(false)
            , m_ReduceFp32ToBf16(false)
            , m_shapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly)
            , m_ImportEnabled(false)
            , m_ModelOptions()
            , m_ProfilingEnabled(false)
            , m_ExportEnabled(false)
            , m_AllowExpandedDims(false)
    {}

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable OptimizerOptionsOpaque instead.", "24.02")
    OptimizerOptions(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16, bool importEnabled,
                     ModelOptions modelOptions = {}, bool exportEnabled = false, bool debugToFile = false)
            : m_ReduceFp32ToFp16(reduceFp32ToFp16)
            , m_Debug(debug)
            , m_DebugToFile(debugToFile)
            , m_ReduceFp32ToBf16(reduceFp32ToBf16)
            , m_shapeInferenceMethod(armnn::ShapeInferenceMethod::ValidateOnly)
            , m_ImportEnabled(importEnabled)
            , m_ModelOptions(modelOptions)
            , m_ProfilingEnabled(false)
            , m_ExportEnabled(exportEnabled)
            , m_AllowExpandedDims(false)
    {
    }

    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use ABI stable OptimizerOptionsOpaque instead.", "24.02")
    OptimizerOptions(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16 = false,
                     ShapeInferenceMethod shapeInferenceMethod = armnn::ShapeInferenceMethod::ValidateOnly,
                     bool importEnabled = false, ModelOptions modelOptions = {}, bool exportEnabled = false,
                     bool debugToFile = false, bool allowExpandedDims = false)
            : m_ReduceFp32ToFp16(reduceFp32ToFp16)
            , m_Debug(debug)
            , m_DebugToFile(debugToFile)
            , m_ReduceFp32ToBf16(reduceFp32ToBf16)
            , m_shapeInferenceMethod(shapeInferenceMethod)
            , m_ImportEnabled(importEnabled)
            , m_ModelOptions(modelOptions)
            , m_ProfilingEnabled(false)
            , m_ExportEnabled(exportEnabled)
            , m_AllowExpandedDims(allowExpandedDims)
    {
    }

    const std::string ToString() const
    {
        std::stringstream stream;
        stream << "OptimizerOptions: \n";
        stream << "\tReduceFp32ToFp16: " << m_ReduceFp32ToFp16 << "\n";
        stream << "\tReduceFp32ToBf16: " << m_ReduceFp32ToBf16 << "\n";
        stream << "\tDebug: " << m_Debug << "\n";
        stream << "\tDebug to file: " << m_DebugToFile << "\n";
        stream << "\tShapeInferenceMethod: " <<
               (m_shapeInferenceMethod == ShapeInferenceMethod::ValidateOnly
               ? "ValidateOnly" : "InferAndValidate") << "\n";
        stream << "\tImportEnabled: " << m_ImportEnabled << "\n";
        stream << "\tExportEnabled: " << m_ExportEnabled << "\n";
        stream << "\tProfilingEnabled: " << m_ProfilingEnabled << "\n";
        stream << "\tAllowExpandedDims: " << m_AllowExpandedDims << "\n";

        stream << "\tModelOptions: \n";
        for (auto optionsGroup : m_ModelOptions)
        {
            for (size_t i=0; i < optionsGroup.GetOptionCount(); i++)
            {
                const armnn::BackendOptions::BackendOption option = optionsGroup.GetOption(i);
                stream << "\t\tBackend: "  << optionsGroup.GetBackendId() << "\n"
                       << "\t\t\tOption: " << option.GetName() << "\n"
                       << "\t\t\tValue: "  << std::string(option.GetValue().ToString()) << "\n";
            }
        }

        return stream.str();
    }

    /// Reduces all Fp32 operators in the model to Fp16 for faster processing.
    /// @Note This feature works best if all operators of the model are in Fp32. ArmNN will add conversion layers
    ///       between layers that weren't in Fp32 in the first place or if the operator is not supported in Fp16.
    ///       The overhead of these conversions can lead to a slower overall performance if too many conversions are
    ///       required.
    bool m_ReduceFp32ToFp16;

    /// Add debug data for easier troubleshooting
    bool m_Debug;

    /// Pass debug data to separate output files for easier troubleshooting
    bool m_DebugToFile;

    /// @Note This feature has been replaced by enabling Fast Math in compute library backend options.
    /// This is currently a placeholder option
    bool m_ReduceFp32ToBf16;

    /// Infer output size when not available
    ShapeInferenceMethod m_shapeInferenceMethod;

    /// Enable Import
    bool m_ImportEnabled;

    /// Enable Model Options
    ModelOptions m_ModelOptions;

    /// Enable profiling dump of the optimizer phase
    bool m_ProfilingEnabled;

    /// Enable Export
    bool m_ExportEnabled;

    /// When calculating tensor sizes, dimensions of size == 1 will be ignored
    bool m_AllowExpandedDims;
};

/// ArmNN performs an optimization on each model/network before it gets loaded for execution. OptimizerOptions provides
/// a set of features that allows the user to customize this optimization on a per model basis.
struct OptimizerOptionsOpaqueImpl;

class OptimizerOptionsOpaque
{
public:
    OptimizerOptionsOpaque();
    OptimizerOptionsOpaque(const OptimizerOptionsOpaque& other);
    ~OptimizerOptionsOpaque();

    OptimizerOptionsOpaque(const OptimizerOptions& OptimizerStruct);

    OptimizerOptionsOpaque& operator=(OptimizerOptionsOpaque other);

    OptimizerOptionsOpaque(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16, bool importEnabled,
                           ModelOptions modelOptions = {}, bool exportEnabled = false, bool debugToFile = false);

    OptimizerOptionsOpaque(bool reduceFp32ToFp16, bool debug, bool reduceFp32ToBf16 = false,
                           ShapeInferenceMethod shapeInferenceMethod = armnn::ShapeInferenceMethod::ValidateOnly,
                           bool importEnabled = false, ModelOptions modelOptions = {}, bool exportEnabled = false,
                           bool debugToFile = false, bool allowExpandedDims = false);

    const std::string ToString() const;

    bool GetProfilingEnabled() const;

    bool GetImportEnabled() const;

    bool GetExportEnabled() const;

    bool GetReduceFp32ToFp16() const;

    bool GetReduceFp32ToBf16() const;

    bool GetDebugEnabled() const;

    bool GetDebugToFileEnabled() const;

    bool GetAllowExpandedDims() const;

    armnn::ModelOptions GetModelOptions() const;

    armnn::ShapeInferenceMethod GetShapeInferenceMethod() const;

    void SetImportEnabled(bool ImportState);

    void SetExportEnabled(bool ExportState);

    void SetProfilingEnabled(bool ProfilingState);

    void SetDebugEnabled(bool DebugState);

    void SetDebugToFileEnabled(bool DebugFileState);

    void SetReduceFp32ToFp16(bool ReduceFp32ToFp16State);

    void SetShapeInferenceMethod(armnn::ShapeInferenceMethod ShapeInferenceMethodType);

    void AddModelOption(armnn::BackendOptions);

    void SetAllowExpandedDims(bool ExpandedDimsAllowed);

private:

    std::unique_ptr<armnn::OptimizerOptionsOpaqueImpl> p_OptimizerOptionsImpl;

};

class IWorkloadFactory;
class NetworkImpl;
using INetworkPtr = std::unique_ptr<INetwork, void(*)(INetwork* network)>;
using IOptimizedNetworkPtr = std::unique_ptr<IOptimizedNetwork, void(*)(IOptimizedNetwork* network)>;

using CompiledBlobDeleter = std::function<void(const void*)>;
using CompiledBlobPtr = std::unique_ptr<void, CompiledBlobDeleter>;

/// Main network class which provides the interface for building up a neural network.
/// This object is subsequently required by the IRuntime::Load() method.
class INetwork
{
public:
    static INetwork* CreateRaw(const NetworkOptions& networkOptions = {});
    static INetworkPtr Create(const NetworkOptions& networkOptions = {});
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
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddConvolution2dLayer(const Convolution2dDescriptor& convolution2dDescriptor,
                                             const char* name = nullptr);

    /// Adds a 3D convolution layer to the network.
    /// @param convolution3dDescriptor - Description of the 3D convolution layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddConvolution3dLayer(const Convolution3dDescriptor& convolution3dDescriptor,
                                             const char* name = nullptr);

    /// Adds a depth to space layer to the network.
    /// @param depthToSpaceDescriptor - Parameters for the depth to space operation.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDepthToSpaceLayer(const DepthToSpaceDescriptor& depthToSpaceDescriptor,
                                            const char* name = nullptr);

    /// Adds a 2D depthwise convolution layer to the network.
    /// @param convolution2dDescriptor - Description of the 2D depthwise convolution layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddDepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& convolution2dDescriptor,
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

    /// Add an ElementwiseBinary layer to the network.
    /// @param name - Optional name for the layer.
    /// @param desc - Descriptor for the elementwiseBinary operations.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddElementwiseBinaryLayer(const ElementwiseBinaryDescriptor& elementwiseUnaryDescriptor,
                                                 const char* name = nullptr);

    /// Add an ElementwiseUnary layer to the network.
    /// @param name - Optional name for the layer.
    /// @param desc - Descriptor for the elementwiseUnary operations.
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
    /// @return - Interface for configuring the layer.
    ///
    /// @note Weights and biases are passed in as inputs. If they are constant tensors you can simply store
    ///       them in a ConstantLayer as seen below. A full example can be found in samples/SimpleSample.cpp.
    ///
    /// @code
    /// // Make sure the IsConstant flag is set on the weightsInfo before passing it to the ConstTensor.
    /// ConstTensor weights(weightsInfo, weightsData);
    ///
    /// // Constant layer that now holds weights data for FullyConnected
    /// IConnectableLayer* const constantWeightsLayer = myNetwork->AddConstantLayer(weights, "weights");
    ///
    /// FullyConnectedDescriptor fullyConnectedDesc;
    /// IConnectableLayer* const fullyConnectedLayer = myNetwork->AddFullyConnectedLayer(fullyConnectedDesc,
    ///                                                                                  "fully connected");
    /// IConnectableLayer* InputLayer = myNetwork->AddInputLayer(0);
    /// InputLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(0));
    /// constantWeightsLayer->GetOutputSlot(0).Connect(fullyConnectedLayer->GetInputSlot(1));
    /// @endcode
    IConnectableLayer* AddFullyConnectedLayer(const FullyConnectedDescriptor& fullyConnectedDescriptor,
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

    /// Adds a 2D pooling layer to the network.
    /// @param pooling2dDescriptor - Pooling2dDescriptor to configure the pooling.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPooling2dLayer(const Pooling2dDescriptor& pooling2dDescriptor,
        const char* name = nullptr);

    /// Adds a 3D pooling layer to the network.
    /// @param pooling3dDescriptor - Pooling3dDescriptor to configure the pooling.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPooling3dLayer(const Pooling3dDescriptor& pooling3dDescriptor,
        const char* name = nullptr);

    /// Adds a Precompiled layer to the network.
    /// Method use is for backend users.
    /// @param preCompiledDescriptor - PreCompiledDescriptor contains parameters for the Precompiled layer.
    /// @param compiledBlobPtr - CompiledBlobPtr pre-compiled object set for the Precompiled layer.
    /// @param backend - optional BackendId set for the Precompiled layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddPrecompiledLayer(const PreCompiledDescriptor& preCompiledDescriptor,
                                           CompiledBlobPtr compiledBlobPtr,
                                           const Optional<BackendId>& backend,
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

    /// Adds an addition layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddAdditionLayer(const char* name = nullptr);

    /// Adds a multiplication layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
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

    /// Adds a shape layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddShapeLayer(const char* name = nullptr);

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
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddDivisionLayer(const char* name = nullptr);

    /// Adds a subtraction layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddSubtractionLayer(const char* name = nullptr);

    /// Add a Maximum layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
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
    ARMNN_DEPRECATED_MSG_REMOVAL_DATE("Use AddElementwiseBinaryLayer instead", "24.02")
    IConnectableLayer* AddMinimumLayer(const char* name = nullptr);

    /// Add Gather layer to the network.
    /// @param descriptor - Description of the gather layer.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddGatherLayer(const GatherDescriptor& descriptor,
                                              const char* name = nullptr);

    /// Add GatherNd layer to the network.
    /// @param name - Optional name for the layer.
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddGatherNdLayer(const char* name = nullptr);

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

    /// Add a UnidirectionalSequenceLstm layer to the network
    /// @param descriptor - Parameters for the UnidirectionalSequenceLstm operation
    /// @param params - Weights and biases for the UnidirectionalSequenceLstm
    /// @param name - Optional name for the layer
    /// @return - Interface for configuring the layer.
    IConnectableLayer* AddUnidirectionalSequenceLstmLayer(const UnidirectionalSequenceLstmDescriptor& descriptor,
                                                          const LstmInputParams& params,
                                                          const char* name = nullptr);

    /// Add a ChannelShuffle layer to the network
    /// @param descriptor - Parameters for the ChannelShuffle operation
    /// @param name - Optional name for the layer
    /// @return - Interface for configuring the layer
    IConnectableLayer* AddChannelShuffleLayer(const ChannelShuffleDescriptor& descriptor,
                                              const char* name = nullptr);

    /// Add a BatchMatMul layer to the network
    /// @param descriptor - Parameters for the BatchMatMul operation
    /// @param name - Optional name for the layer
    /// @return - Interface for configuring the layer
    IConnectableLayer* AddBatchMatMulLayer(const BatchMatMulDescriptor& descriptor,
                                           const char* name = nullptr);

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
    friend IOptimizedNetworkPtr Optimize(const INetwork& network,
                                         const std::vector<BackendId>& backendPreferences,
                                         const IDeviceSpec& deviceSpec,
                                         const OptimizerOptionsOpaque& options,
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
class IProfiler;
class IOptimizedNetwork
{
public:
    static void Destroy(IOptimizedNetwork* network);

    Status PrintGraph();
    Status SerializeToDot(std::ostream& stream) const;

    arm::pipe::ProfilingGuid GetGuid() const;

    size_t GetNumInputs() const;
    size_t GetNumOutputs() const;

    void ExecuteStrategy(IStrategy& strategy) const;

    /// Creates a copy of the IOptimizedNetwork. The IOptimizedNetwork will not be reoptimized,
    /// the provided ModelOptions will only be used when creating a LoadedNetwork.
    IOptimizedNetwork(const IOptimizedNetwork& other, const ModelOptions& modelOptions);
    IOptimizedNetwork(std::unique_ptr<Graph> graph);
    IOptimizedNetwork(std::unique_ptr<OptimizedNetworkImpl> impl);
    ~IOptimizedNetwork();

    const std::shared_ptr<IProfiler>& GetProfiler() const;

protected:
    friend class LoadedNetwork;

    friend class experimental::AsyncNetworkImpl;
    friend class experimental::WorkingMemHandle;

    friend Graph& GetGraphForTesting(IOptimizedNetwork* optNetPtr);
    friend ModelOptions& GetModelOptionsForTesting(IOptimizedNetwork* optNetPtr);
    friend IOptimizedNetworkPtr Optimize(const INetwork& inNetwork,
                                         const std::vector<BackendId>& backendPreferences,
                                         const IDeviceSpec& deviceSpec,
                                         const OptimizerOptionsOpaque& options,
                                         Optional<std::vector<std::string>&> messages);
    friend IOptimizedNetworkPtr Optimize(const Graph& inGraph,
                                         const std::vector<BackendId>& backendPreferences,
                                         const IDeviceSpec& deviceSpec,
                                         const OptimizerOptionsOpaque& options,
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
                              const OptimizerOptionsOpaque& options = OptimizerOptionsOpaque(),
                              Optional<std::vector<std::string>&> messages = EmptyOptional());

/// Create an optimized version of the network
/// @param inGraph Graph to be optimized.
/// @param backendPreferences The choice of the backend ordered by user preferences.
/// @param deviceSpec DeviceSpec object as queried from the runtime. See IRuntime::GetDeviceSpec()
/// @param messages If there are failures or warnings a string describing same will be added to the vector
/// @param options OptimizerOptions object with optimizer configuration options
/// @return An IOptimizedNetworkPtr interface to the optimized network, throws an exception derived from
/// armnn::Exception if process fails.

IOptimizedNetworkPtr Optimize(const Graph& inGraph,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptionsOpaque& options,
                              Optional<std::vector<std::string>&> messages = EmptyOptional());

/// Accept legacy OptimizerOptions
IOptimizedNetworkPtr Optimize(const Graph& inGraph,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options,
                              Optional<std::vector<std::string>&> messages = EmptyOptional());

/// Accept legacy OptimizerOptions
IOptimizedNetworkPtr Optimize(const INetwork& network,
                              const std::vector<BackendId>& backendPreferences,
                              const IDeviceSpec& deviceSpec,
                              const OptimizerOptions& options,
                              Optional<std::vector<std::string>&> messages = EmptyOptional());

} //namespace armnn
