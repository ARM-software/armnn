//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include "Layers.hpp"
#include "Graph.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/Workload.hpp"
#include "backends/WorkloadFactory.hpp"

#include "Permute.hpp"

#include <queue>


namespace armnn
{

template <typename LayerType, typename ... Params>
LayerType* Layer::CloneBase(Graph& graph, Params&& ... params) const
{
    LayerType* const layer = graph.AddLayer<LayerType>(std::forward<Params>(params)...);

    layer->SetComputeDevice(m_ComputeDevice);
    layer->SetGuid(GetGuid());

    return layer;
}

ActivationLayer::ActivationLayer(const ActivationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Activation, param, name)
{
}

std::unique_ptr<IWorkload> ActivationLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    ActivationQueueDescriptor descriptor;
    return factory.CreateActivation(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ActivationLayer* ActivationLayer::Clone(Graph& graph) const
{
    return CloneBase<ActivationLayer>(graph, m_Param, GetName());
}

void ActivationLayer::ValidateTensorShapesFromInputs()
{
    auto& info = GetInputSlot(0).GetConnection()->GetTensorInfo();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(info.GetShape()),
                     "ActivationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

AdditionLayer::AdditionLayer(const char* name)
    : Layer(2, 1, LayerType::Addition, name)
{
}

std::unique_ptr<IWorkload> AdditionLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    AdditionQueueDescriptor descriptor;
    return factory.CreateAddition(descriptor, PrepInfoAndDesc(descriptor, graph));
}

AdditionLayer* AdditionLayer::Clone(Graph& graph) const
{
    return CloneBase<AdditionLayer>(graph, GetName());
}

void AdditionLayer::ValidateTensorShapesFromInputs()
{
    auto& input0 = GetInputSlot(0).GetConnection()->GetTensorInfo();
    auto& input1 = GetInputSlot(1).GetConnection()->GetTensorInfo();

    // Get the max of the inputs
    BOOST_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();
    std::vector<unsigned int> dims(numDims);

    // validate inputs are broadcast compatible
#if !NDEBUG
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        if (dim0 != dim1)
        {
            BOOST_ASSERT_MSG(dim0 == 1 || dim1 == 1, "Dimensions should either match or one should be of size 1.");
        }
    }
#endif

    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        dims[i] = std::max(dim0, dim1);
    }

    TensorShape outShape(numDims, dims.data());
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "AdditionLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

BatchNormalizationLayer::BatchNormalizationLayer(const armnn::BatchNormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::BatchNormalization, param, name)
{
}

std::unique_ptr<IWorkload> BatchNormalizationLayer::CreateWorkload(const Graph& graph,
                                                                   const IWorkloadFactory& factory) const
{
    BatchNormalizationQueueDescriptor descriptor;

    descriptor.m_Mean = m_Mean.get();
    descriptor.m_Variance = m_Variance.get();
    descriptor.m_Beta = m_Beta.get();
    descriptor.m_Gamma = m_Gamma.get();
    return factory.CreateBatchNormalization(descriptor, PrepInfoAndDesc(descriptor, graph));
}

BatchNormalizationLayer* BatchNormalizationLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<BatchNormalizationLayer>(graph, m_Param, GetName());

    layer->m_Mean = m_Mean ? std::make_unique<ScopedCpuTensorHandle>(*m_Mean) : nullptr;
    layer->m_Variance = m_Variance ? std::make_unique<ScopedCpuTensorHandle>(*m_Variance) : nullptr;
    layer->m_Beta = m_Beta ? std::make_unique<ScopedCpuTensorHandle>(*m_Beta) : nullptr;
    layer->m_Gamma = m_Gamma ? std::make_unique<ScopedCpuTensorHandle>(*m_Gamma) : nullptr;

    return std::move(layer);
}

void BatchNormalizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "BatchNormalizationLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "BatchNormalizationLayer: TensorInfo must be set on connected OutputSlot.");

    auto& info = GetInputSlot(0).GetConnection()->GetTensorInfo();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(info.GetShape()),
                     "BatchNormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

Convolution2dLayer::Convolution2dLayer(const Convolution2dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Convolution2d, param, name)
{
}

std::unique_ptr<IWorkload> Convolution2dLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    Convolution2dQueueDescriptor descriptor;

    descriptor.m_Weight = m_Weight.get();
    if (m_Param.m_BiasEnabled)
    {
        descriptor.m_Bias = m_Bias.get();
    }
    return factory.CreateConvolution2d(descriptor, PrepInfoAndDesc(descriptor, graph));
}

Convolution2dLayer* Convolution2dLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<Convolution2dLayer>(graph, m_Param, GetName());
    layer->m_Weight = m_Weight ? std::make_unique<ScopedCpuTensorHandle>(*m_Weight) : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? std::make_unique<ScopedCpuTensorHandle>(*m_Bias) : nullptr;
    }

    return std::move(layer);
}

void Convolution2dLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "Convolution2dLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "Convolution2dLayer: TensorInfo must be set on connected OutputSlot.");


    IOutputSlot* input = GetInputSlot(0).GetConnection();
    const TensorShape& inputShape = input->GetTensorInfo().GetShape();
    const TensorShape filterShape = m_Weight->GetTensorInfo().GetShape();

    // If we support multiple batch dimensions in the future, then this assert will need to change.
    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Convolutions will always have 4D input.");

    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inBatchSize = inputShape[0];

    unsigned int filterWidth = filterShape[3];
    unsigned int readWidth = (inWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - (filterWidth);
    unsigned int outWidth =  1+(readWidth / m_Param.m_StrideX);

    unsigned int filterHeight = filterShape[2];
    unsigned int readHeight = (inHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - (filterHeight);
    unsigned int outHeight = 1+(readHeight / m_Param.m_StrideY);

    unsigned int outChannels = filterShape[0];
    unsigned int outBatchSize = inBatchSize;

    TensorShape shapeOut({outBatchSize, outChannels, outHeight, outWidth});
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(shapeOut),
                     "Convolution2dLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}


DepthwiseConvolution2dLayer::DepthwiseConvolution2dLayer(const DepthwiseConvolution2dDescriptor& param,
                                                         const char* name)
    : LayerWithParameters(1, 1, LayerType::DepthwiseConvolution2d, param, name)
{
}

std::unique_ptr<IWorkload> DepthwiseConvolution2dLayer::CreateWorkload(const Graph&                  graph,
                                                                       const IWorkloadFactory& factory) const
{
    DepthwiseConvolution2dQueueDescriptor descriptor;

    descriptor.m_Weight = m_Weight.get();
    if (m_Param.m_BiasEnabled)
    {
        descriptor.m_Bias = m_Bias.get();
    }
    return factory.CreateDepthwiseConvolution2d(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DepthwiseConvolution2dLayer* DepthwiseConvolution2dLayer::Clone(Graph& graph) const
{
    auto layer      = CloneBase<DepthwiseConvolution2dLayer>(graph, m_Param, GetName());
    layer->m_Weight = m_Weight ? std::make_unique<ScopedCpuTensorHandle>(*m_Weight) : nullptr;

    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? std::make_unique<ScopedCpuTensorHandle>(*m_Bias) : nullptr;
    }

    return std::move(layer);
}

void DepthwiseConvolution2dLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "DepthwiseConvolution2dLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "DepthwiseConvolution2dLayer: TensorInfo must be set on connected OutputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();
    const TensorShape& inputShape = input->GetTensorInfo().GetShape();
    const TensorShape filterShape = m_Weight->GetTensorInfo().GetShape();

    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Convolutions will always have 4D input.");

    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inBatchSize = inputShape[0];

    unsigned int filterWidth = filterShape[3];
    unsigned int readWidth = (inWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - (filterWidth);
    unsigned int outWidth =  1+(readWidth / m_Param.m_StrideX);

    unsigned int filterHeight = filterShape[2];
    unsigned int readHeight = (inHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - (filterHeight);
    unsigned int outHeight = 1+(readHeight / m_Param.m_StrideY);
    unsigned int depthMultiplier = filterShape[0];

    unsigned int outChannels = filterShape[1]*depthMultiplier;
    unsigned int outBatchSize = inBatchSize;

    TensorShape outShape({outBatchSize, outChannels, outHeight, outWidth});
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "DepthwiseConvolution2dLayer: "
                         "TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

FakeQuantizationLayer::FakeQuantizationLayer(const FakeQuantizationDescriptor& param, const char* name)
: LayerWithParameters(1, 1, LayerType::FakeQuantization, param, name)
{
}

std::unique_ptr<IWorkload> FakeQuantizationLayer::CreateWorkload(const Graph& graph,
                                                                const IWorkloadFactory& factory) const
{
    FakeQuantizationQueueDescriptor descriptor;
    return factory.CreateFakeQuantization(descriptor, PrepInfoAndDesc(descriptor, graph) );
}

FakeQuantizationLayer* FakeQuantizationLayer::Clone(Graph& graph) const
{
    return CloneBase<FakeQuantizationLayer>(graph, m_Param, GetName());
}

void FakeQuantizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "FakeQuantizationLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "FakeQuantizationLayer: TensorInfo must be set on connected OutputSlot.");


    IOutputSlot* input = GetInputSlot(0).GetConnection();

    // input and output shapes are the same
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "FakeQuantizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

FloorLayer::FloorLayer(const char* name)
 : Layer(1, 1, LayerType::Floor, name)
{
}

std::unique_ptr<IWorkload> FloorLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    FloorQueueDescriptor descriptor;
    return factory.CreateFloor(descriptor, PrepInfoAndDesc(descriptor, graph));
}

FloorLayer* FloorLayer::Clone(Graph& graph) const
{
    return CloneBase<FloorLayer>(graph, GetName());
}

void FloorLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "FloorLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "FloorLayer: TensorInfo must be set on connected OutputSlot.");

    // input and output shapes are the same
    IOutputSlot* input = GetInputSlot(0).GetConnection();
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "FloorLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

FullyConnectedLayer::FullyConnectedLayer(const FullyConnectedDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::FullyConnected, param, name)
{
}

std::unique_ptr<IWorkload> FullyConnectedLayer::CreateWorkload(const Graph& graph,
                                                               const IWorkloadFactory& factory) const
{
    FullyConnectedQueueDescriptor descriptor;

    descriptor.m_Weight = m_Weight.get();
    if (m_Param.m_BiasEnabled)
    {
        descriptor.m_Bias = m_Bias.get();
    }
    return factory.CreateFullyConnected(descriptor, PrepInfoAndDesc(descriptor, graph));
}

FullyConnectedLayer* FullyConnectedLayer::Clone(Graph& graph) const
{
    auto layer = CloneBase<FullyConnectedLayer>(graph, m_Param, GetName());

    layer->m_Weight = m_Weight ? std::make_unique<ScopedCpuTensorHandle>(*m_Weight) : nullptr;
    if (layer->m_Param.m_BiasEnabled)
    {
        layer->m_Bias = m_Bias ? std::make_unique<ScopedCpuTensorHandle>(*m_Bias) : nullptr;
    }

    return std::move(layer);
}

void FullyConnectedLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "FullyConnectedLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "FullyConnectedLayer: TensorInfo must be set on connected OutputSlot.");


    TensorShape const& weightShape = m_Weight->GetTensorInfo().GetShape();

    // output for FC is [1, w[1]]
    unsigned int batches = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape()[0];
    unsigned int dimIdx = m_Param.m_TransposeWeightMatrix ? 0 : 1;
    TensorShape outShape({batches, weightShape[dimIdx]});

    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "FullyConnectedLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

InputLayer::InputLayer(LayerBindingId id, const char* name)
    : BindableLayer(0, 1, LayerType::Input, name, id)
{
}

std::unique_ptr<IWorkload> InputLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    return nullptr;
}

InputLayer* InputLayer::Clone(Graph& graph) const
{
    return CloneBase<InputLayer>(graph, GetBindingId(), GetName());
}

void InputLayer::ValidateTensorShapesFromInputs()
{
    //The input layer should already have it's inputs set during graph building phase in the driver/parser.
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).IsTensorInfoSet(),
                                               "InputLayer should already have the TensorInfo set.");
}


MergerLayer::MergerLayer(const OriginsDescriptor& param, const char* name)
    : LayerWithParameters(param.GetNumViews(), 1, LayerType::Merger, param, name)
{
}

std::unique_ptr<IWorkload> MergerLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    MergerQueueDescriptor descriptor;

    // copy the view origins to the descriptor
    descriptor.m_ViewOrigins.reserve(m_Param.GetNumViews());
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }

    return factory.CreateMerger(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void MergerLayer::CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory)
{
    //if sub tensors are supported than the merger
    //just needs to make sure that the outputs of the prev layer
    //are made subtensors of the output of the merger layer
    m_OutputHandlers[0].CreateTensorHandles(factory);
    if (factory.SupportsSubTensors())
    {
        std::queue<MergerLayer*> m_MergerLayers;

        m_MergerLayers.push(this);
        while (!m_MergerLayers.empty())
        {
            MergerLayer* currentLayer = m_MergerLayers.front();
            ITensorHandle* parentTensor = currentLayer->GetOutputHandler(0).GetData();

            m_MergerLayers.pop();

            const unsigned int numInputSlots = currentLayer->GetNumInputSlots();
            for (unsigned int i = 0; i < numInputSlots; ++i)
            {
                OutputSlot* slot = currentLayer->GetInputSlot(i).GetConnectedOutputSlot();
                OutputHandler& outputHandler = slot->GetOutputHandler();
                outputHandler.SetData(factory.CreateSubTensorHandle(*parentTensor,
                                                                    outputHandler.GetTensorInfo().GetShape(),
                                                                    currentLayer->m_Param.GetViewOrigin(i)));

                Layer& inputLayer = slot->GetOwningLayer();
                if (inputLayer.GetType() == LayerType::Merger)
                {
                    m_MergerLayers.push(boost::polymorphic_downcast<MergerLayer*>(&inputLayer));
                }
            }
        }
    }
}

MergerLayer* MergerLayer::Clone(Graph& graph) const
{
    return CloneBase<MergerLayer>(graph, m_Param, GetName());
}

void MergerLayer::ValidateTensorShapesFromInputs()
{
    // Validate Merger layer
    ConditionalThrow<LayerValidationException>(m_Param.GetNumViews() == GetNumInputSlots(),
                     "MergerLayer: Num Inputs must match num views.");

    unsigned int numDims = m_Param.GetNumDimensions();
    for (unsigned int i=0; i<GetNumInputSlots(); i++)
    {
        auto& inputInfo = GetInputSlot(i).GetConnection()->GetTensorInfo();

        boost::ignore_unused(inputInfo);
        ConditionalThrow<LayerValidationException>(numDims == inputInfo.GetNumDimensions(),
                         "MergerLayer: Num Dimensions must match all inputs.");
    }

    // Find the bounding box (extents) of all the views
    std::vector<unsigned int> extentMin(numDims);
    std::vector<unsigned int> extentMax(numDims);
    for (unsigned int i = 0; i < GetNumInputSlots(); i++)
    {
        const uint32_t* origin = m_Param.GetViewOrigin(i);
        const armnn::TensorShape& shape = GetInputSlot(i).GetConnection()->GetTensorInfo().GetShape();
        for (unsigned int d = 0; d < numDims; d++)
        {
            extentMin[d] = std::min(extentMin[d], origin[d]);
            extentMax[d] = std::max(extentMax[d], origin[d] + shape[d]);
        }
    }

    // Check that the bounding box starts at the origin
    if (!std::all_of(extentMin.begin(), extentMin.end(), [](unsigned int s) { return s == 0; }))
    {
        throw LayerValidationException("MergerLayer: there is no view that starts at the origin");
    }

    // Check that there are no overlaps of views (this would lead to undefined output at those locations).
    // Check each pair of views against each other
    // (and don't bother to check against self, or check the same pair both ways round)
    for (unsigned int a = 0; a < GetNumInputSlots(); a++)
    {
        const uint32_t* aOrigin = m_Param.GetViewOrigin(a);
        const armnn::TensorShape& aShape = GetInputSlot(a).GetConnection()->GetTensorInfo().GetShape();
        for (unsigned int b = 0; b < a; b++)
        {
            const uint32_t* bOrigin = m_Param.GetViewOrigin(b);
            const armnn::TensorShape& bShape = GetInputSlot(b).GetConnection()->GetTensorInfo().GetShape();

            bool allAxesOverlap = true;
            for (unsigned int d = 0; d < numDims && allAxesOverlap; d++)
            {
                unsigned int a1 = aOrigin[d];
                unsigned int a2 = aOrigin[d] + aShape[d];

                unsigned int b1 = bOrigin[d];
                unsigned int b2 = bOrigin[d] + bShape[d];

                if (a2 <= b1 || b2 <= a1)
                {
                    allAxesOverlap = false;
                }
            }
            if (allAxesOverlap)
            {
                throw LayerValidationException("MergerLayer: Some views overlap.");
            }
        }
    }

    // Check that there are no "holes", i.e. regions of the output which is not covered by a view.
    // Because we already checked that there are no overlaps, this can be done simply by checking that
    // the total 'volume' of the views is the same as the output.
    unsigned int totalViewsVolume = 0;
    for (unsigned int i = 0; i < GetNumInputSlots(); i++)
    {
        totalViewsVolume += GetInputSlot(i).GetConnection()->GetTensorInfo().GetNumElements();
    }
    unsigned int outputVolume = 1;
    for (unsigned int d = 0; d < numDims; d++)
    {
        outputVolume *= (extentMax[d] - extentMin[d]);
    }
    if (totalViewsVolume != outputVolume)
    {
        throw LayerValidationException("MergerLayer: there are some gaps between views");
    }

    TensorShape outShape(numDims, extentMax.data());
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "MergerLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

MultiplicationLayer::MultiplicationLayer(const char* name)
    : Layer(2, 1, LayerType::Multiplication, name)
{
}

std::unique_ptr<IWorkload> MultiplicationLayer::CreateWorkload(const Graph&            graph,
                                                               const IWorkloadFactory& factory) const
{
    MultiplicationQueueDescriptor descriptor;

    return factory.CreateMultiplication(descriptor, PrepInfoAndDesc(descriptor, graph));
}

MultiplicationLayer* MultiplicationLayer::Clone(Graph& graph) const
{
    return CloneBase<MultiplicationLayer>(graph, GetName());
}

void MultiplicationLayer::ValidateTensorShapesFromInputs()
{
    auto& input0 = GetInputSlot(0).GetConnection()->GetTensorInfo();
    auto& input1 = GetInputSlot(1).GetConnection()->GetTensorInfo();

    // Get the max of the inputs
    BOOST_ASSERT(input0.GetNumDimensions() == input1.GetNumDimensions());
    unsigned int numDims = input0.GetNumDimensions();
    std::vector<unsigned int> dims(numDims);

    // validate inputs are broadcast compatible
#if !NDEBUG
    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        if (dim0 != dim1)
        {
            BOOST_ASSERT_MSG(dim0 == 1 || dim1 == 1, "Dimensions should either match or one should be of size 1.");
        }
    }
#endif

    for (unsigned int i = 0; i < numDims; i++)
    {
        unsigned int dim0 = input0.GetShape()[i];
        unsigned int dim1 = input1.GetShape()[i];
        dims[i] = std::max(dim0, dim1);
    }

    TensorShape outShape(numDims, dims.data());
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "MultiplicationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

NormalizationLayer::NormalizationLayer(const NormalizationDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Normalization, param, name)
{
}

std::unique_ptr<IWorkload> NormalizationLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    NormalizationQueueDescriptor descriptor;
    return factory.CreateNormalization(descriptor, PrepInfoAndDesc(descriptor, graph));
}

NormalizationLayer* NormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<NormalizationLayer>(graph, m_Param, GetName());
}

void NormalizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "NormalizationLayer: Input slot must be connected.");

    const TensorShape& outShape = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "NormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

OutputLayer::OutputLayer(LayerBindingId id, const char* name)
    : BindableLayer(1, 0, LayerType::Output, name, id)
{
}

std::unique_ptr<IWorkload> OutputLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    return nullptr;
}

OutputLayer* OutputLayer::Clone(Graph& graph) const
{
    return CloneBase<OutputLayer>(graph, GetBindingId(), GetName());
}

void OutputLayer::ValidateTensorShapesFromInputs()
{
    // Just validate the input is connected
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "OutputLayer: Input slot must be connected.");
}

PermuteLayer::PermuteLayer(const PermuteDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Permute, param, name)
{
}

std::unique_ptr<IWorkload> PermuteLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    PermuteQueueDescriptor descriptor;
    return factory.CreatePermute(descriptor, PrepInfoAndDesc(descriptor, graph));
}

PermuteLayer* PermuteLayer::Clone(Graph& graph) const
{
    return CloneBase<PermuteLayer>(graph, m_Param, GetName());
}

void PermuteLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "PermuteLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "PermuteLayer: TensorInfo must be set on connected InputSlot.");

    const TensorInfo& infoIn = GetInputSlot(0).GetConnection()->GetTensorInfo();
    TensorShape shapeOut = armnnUtils::Permuted(infoIn.GetShape(), m_Param.m_DimMappings);
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(shapeOut),
                     "PermuteLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

Pooling2dLayer::Pooling2dLayer(const Pooling2dDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Pooling2d, param, name)
{
}

std::unique_ptr<IWorkload> Pooling2dLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    Pooling2dQueueDescriptor descriptor;
    return factory.CreatePooling2d(descriptor, PrepInfoAndDesc(descriptor, graph));
}

Pooling2dLayer* Pooling2dLayer::Clone(Graph& graph) const
{
    return CloneBase<Pooling2dLayer>(graph, m_Param, GetName());
}

void Pooling2dLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "Pooling2dLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "Pooling2dLayer: TensorInfo must be set on connected InputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();
    const TensorShape& inputShape = input->GetTensorInfo().GetShape();

    // If we support multiple batch dimensions in the future, then this assert will need to change.
    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Pooling2dLayer will always have 4D input.");


    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inChannels = inputShape[1];
    unsigned int inBatchSize = inputShape[0];

    bool isGlobalPooling = (m_Param.m_StrideX==0 && m_Param.m_StrideY==0);
    unsigned int outWidth = 1;
    unsigned int outHeight = 1;
    if (!isGlobalPooling)
    {
        BOOST_ASSERT_MSG(m_Param.m_StrideX!=0 && m_Param.m_StrideY!=0,
                         "Stride can only be zero when performing global pooling");

        auto CalcSize = [](auto inSize, auto lowPad, auto highPad, auto poolSize, auto stride, auto padMethod,
                           auto outputShapeRounding)
            {
                unsigned int readSize = inSize + lowPad + highPad - poolSize;
                float div = static_cast<float>(readSize) / static_cast<float>(stride);

                unsigned int size = 0;
                switch (outputShapeRounding)
                {
                    case OutputShapeRounding::Ceiling:
                        size = static_cast<unsigned int>(ceil(div)) + 1;
                        break;
                    case OutputShapeRounding ::Floor:
                        size = static_cast<unsigned int>(floor(div)) + 1;
                        break;
                    default:
                        BOOST_ASSERT_MSG(false, "Unsupported Output Shape Rounding");
                }

                // Make sure that border operations will start from inside the input and not the padded area
                // This is what both Caffe and CL does...
                if ((size - 1)*stride >= inSize + lowPad)
                {
                    --size;
                }

                return size;
            };

        outWidth = CalcSize(inWidth, m_Param.m_PadLeft, m_Param.m_PadRight, m_Param.m_PoolWidth, m_Param.m_StrideX,
                            m_Param.m_PaddingMethod, m_Param.m_OutputShapeRounding);
        outHeight= CalcSize(inHeight, m_Param.m_PadTop, m_Param.m_PadBottom, m_Param.m_PoolHeight, m_Param.m_StrideY,
                            m_Param.m_PaddingMethod, m_Param.m_OutputShapeRounding);


    }
    unsigned int outChannels = inChannels;
    unsigned int outBatchSize = inBatchSize;

    TensorShape shapeOut({outBatchSize, outChannels, outHeight, outWidth});

    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(shapeOut),
               "Pooling2dLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

SoftmaxLayer::SoftmaxLayer(const SoftmaxDescriptor &param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Softmax, param, name)
{
}

std::unique_ptr<IWorkload> SoftmaxLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    SoftmaxQueueDescriptor descriptor;
    return factory.CreateSoftmax(descriptor, PrepInfoAndDesc(descriptor, graph));
}

SoftmaxLayer* SoftmaxLayer::Clone(Graph& graph) const
{
    return CloneBase<SoftmaxLayer>(graph, m_Param, GetName());
}

void SoftmaxLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "SoftmaxLayer: Input slot must be connected.");
    const TensorShape& outShape = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "SoftmaxLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

SplitterLayer::SplitterLayer(const ViewsDescriptor& param, const char* name)
    : LayerWithParameters(1, param.GetNumViews(), LayerType::Splitter, param, name)
{
}

std::unique_ptr<IWorkload> SplitterLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    SplitterQueueDescriptor descriptor;

    // copy the window origins to the descriptor
    for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
    {
        descriptor.m_ViewOrigins.emplace_back(
            std::vector<unsigned int>(m_Param.GetViewOrigin(i), m_Param.GetViewOrigin(i) + m_Param.GetNumDimensions()));
    }

    return factory.CreateSplitter(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void SplitterLayer::CreateTensorHandles(Graph& graph, const IWorkloadFactory& factory)
{
    //if sub tensors are supported than all the "splitter" need to do is to
    //set the outputs to be appropriate sub tensors of the input.
    if (factory.SupportsSubTensors())
    {
        const OutputHandler& outputHandler = GetInputSlots()[0].GetConnectedOutputSlot()->GetOutputHandler();

        ITensorHandle* inputData = outputHandler.GetData();
        //create the outputs as subtensors of the input
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            m_OutputHandlers[i].SetData(factory.CreateSubTensorHandle(*inputData,
                                                                      m_OutputHandlers[i].GetTensorInfo().GetShape(),
                                                                      m_Param.GetViewOrigin(i)));
        }
    }
    else
    {
        for (unsigned int i = 0; i < m_Param.GetNumViews(); ++i)
        {
            m_OutputHandlers[i].CreateTensorHandles(factory);
        }
    }
}

SplitterLayer* SplitterLayer::Clone(Graph& graph) const
{
    return CloneBase<SplitterLayer>(graph, m_Param, GetName());
}

void SplitterLayer::ValidateTensorShapesFromInputs()
{
    //Output shapes must match View shapes.
    for (unsigned int viewIdx = 0; viewIdx < m_Param.GetNumViews(); viewIdx++)
    {
        const uint32_t* sizes = m_Param.GetViewSizes(viewIdx);

        TensorShape outShape(m_Param.GetNumDimensions(), sizes);
        ConditionalThrow<LayerValidationException>(GetOutputSlot(viewIdx).ValidateTensorShape(outShape),
                         "SplitterLayer: View sizes must match output tensor shapes.");
    }
}

MemCopyLayer::MemCopyLayer(const char* name)
    : Layer(1, 1, LayerType::MemCopy, name)
{
}

MemCopyLayer* MemCopyLayer::Clone(Graph& graph) const
{
    return CloneBase<MemCopyLayer>(graph, GetName());
}

std::unique_ptr<IWorkload> MemCopyLayer::CreateWorkload(const Graph& graph, const IWorkloadFactory& factory) const
{
    MemCopyQueueDescriptor descriptor;
    return factory.CreateMemCopy(descriptor, PrepInfoAndDesc(descriptor, graph));
}

void MemCopyLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "MemCopyLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "MemCopyLayer: TensorInfo must be set on connected OutputSlot.");


    IOutputSlot* input = GetInputSlot(0).GetConnection();

    // input and output shapes are the same
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "MemCopyLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

ResizeBilinearLayer::ResizeBilinearLayer(const ResizeBilinearDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::ResizeBilinear, param, name)
{
}

std::unique_ptr<IWorkload> ResizeBilinearLayer::CreateWorkload(const Graph& graph,
                                                               const IWorkloadFactory& factory) const
{
    ResizeBilinearQueueDescriptor descriptor;
    return factory.CreateResizeBilinear(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ResizeBilinearLayer* ResizeBilinearLayer::Clone(Graph& graph) const
{
    return CloneBase<ResizeBilinearLayer>(graph, m_Param, GetName());
}

void ResizeBilinearLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "MemCopyLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "MemCopyLayer: TensorInfo must be set on connected OutputSlot.");

    const TensorShape& inputShape = GetInputSlot(0).GetConnection()->GetTensorInfo().GetShape();
    unsigned int outWidth = m_Param.m_TargetWidth;
    unsigned int outHeight = m_Param.m_TargetHeight;
    unsigned int outChannels = inputShape[1];
    unsigned int outBatch = inputShape[0];
    TensorShape outShape({outBatch, outChannels, outHeight, outWidth});
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "ResizeBilinearLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

L2NormalizationLayer::L2NormalizationLayer(const char* name)
    : Layer(1, 1, LayerType::L2Normalization, name)
{
}

std::unique_ptr<IWorkload> L2NormalizationLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    L2NormalizationQueueDescriptor descriptor;
    return factory.CreateL2Normalization(descriptor, PrepInfoAndDesc(descriptor, graph));
}

L2NormalizationLayer* L2NormalizationLayer::Clone(Graph& graph) const
{
    return CloneBase<L2NormalizationLayer>(graph, GetName());
}

void L2NormalizationLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "L2NormalizationLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "L2NormalizationLayer: TensorInfo must be set on connected OutputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();

    // input and output shapes are the same
    TensorShape const& outShape = input->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "L2NormalizationLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

ConstantLayer::ConstantLayer(const std::shared_ptr<ScopedCpuTensorHandle>& input, const char* name)
    : Layer(0, 1, LayerType::Constant, name)
    , m_LayerOutput(input)
{
}

std::unique_ptr<IWorkload> ConstantLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ConstantQueueDescriptor descriptor;
    descriptor.m_LayerOutput = m_LayerOutput.get();
    return factory.CreateConstant(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ConstantLayer* ConstantLayer::Clone(Graph& graph) const
{
    // Cloned layers share the same layer output object
    return CloneBase<ConstantLayer>(graph, m_LayerOutput, GetName());
}

void ConstantLayer::ValidateTensorShapesFromInputs()
{
    // get the output shape from the value of the constant layer
    TensorShape const& outShape = m_LayerOutput->GetTensorInfo().GetShape();
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                     "ConstantLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

ReshapeLayer::ReshapeLayer(const ReshapeDescriptor& param, const char* name)
    : LayerWithParameters(1, 1, LayerType::Reshape, param, name)
{
}

std::unique_ptr<IWorkload> ReshapeLayer::CreateWorkload(const Graph& graph,
    const IWorkloadFactory& factory) const
{
    ReshapeQueueDescriptor descriptor;
    return factory.CreateReshape(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ReshapeLayer* ReshapeLayer::Clone(Graph& graph) const
{
    return CloneBase<ReshapeLayer>(graph, m_Param, GetName());
}

void ReshapeLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                     "ReshapeLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                     "ReshapeLayer: TensorInfo must be set on connected OutputSlot.");
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(m_Param.m_TargetShape),
                     "ReshapeLayer: TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

std::unique_ptr<IWorkload> DetectionOutputLayer::CreateWorkload(const Graph& graph,
                                                  const IWorkloadFactory& factory) const
{
    DetectionOutputQueueDescriptor descriptor;
    return factory.CreateReshape(descriptor, PrepInfoAndDesc(descriptor, graph));
}

DetectionOutputLayer* DetectionOutputLayer::Clone(Graph& graph) const
{
    return CloneBase<DetectionOutputLayer>(graph, m_Param, GetName());
}

void DetectionOutputLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "DetectionOutputLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                                               "DetectionOutputLayer: TensorInfo must be set on connected OutputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();
    const TensorShape& inputShape = input->GetTensorInfo().GetShape();
    //const TensorShape filterShape = m_Weight->GetTensorInfo().GetShape();

    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Convolutions will always have 4D input.");

    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inBatchSize = inputShape[0];

    /*unsigned int filterWidth = filterShape[3];
    unsigned int readWidth = (inWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - (filterWidth);
    unsigned int outWidth =  1+(readWidth / m_Param.m_StrideX);

    unsigned int filterHeight = filterShape[2];
    unsigned int readHeight = (inHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - (filterHeight);
    unsigned int outHeight = 1+(readHeight / m_Param.m_StrideY);
    unsigned int depthMultiplier = filterShape[0];*/

    unsigned int outChannels = filterShape[1]*depthMultiplier;
    unsigned int outBatchSize = inBatchSize;

    TensorShape outShape({outBatchSize, outChannels, outHeight, outWidth});
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                                               "DetectionOutputLayer: "
                                                       "TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

std::unique_ptr<IWorkload> ReorgLayer::CreateWorkload(const Graph& graph,
                                                  const IWorkloadFactory& factory) const
{
    ReorgQueueDescriptor descriptor;
    return factory.CreateReshape(descriptor, PrepInfoAndDesc(descriptor, graph));
}

ReorgLayer* ReorgLayer::Clone(Graph& graph) const
{
    return CloneBase<ReorgLayer>(graph, m_Param, GetName());
}

void ReorgLayer::ValidateTensorShapesFromInputs()
{
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection() != nullptr,
                                               "ReorgLayer: InputSlot must be connected to an OutputSlot");
    ConditionalThrow<LayerValidationException>(GetInputSlot(0).GetConnection()->IsTensorInfoSet(),
                                               "ReorgLayer: TensorInfo must be set on connected OutputSlot.");

    IOutputSlot* input = GetInputSlot(0).GetConnection();
    const TensorShape& inputShape = input->GetTensorInfo().GetShape();
    const TensorShape filterShape = m_Weight->GetTensorInfo().GetShape();

    BOOST_ASSERT_MSG(inputShape.GetNumDimensions() == 4, "Convolutions will always have 4D input.");

    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inBatchSize = inputShape[0];

    /*unsigned int filterWidth = filterShape[3];
    unsigned int readWidth = (inWidth + m_Param.m_PadLeft + m_Param.m_PadRight) - (filterWidth);
    unsigned int outWidth =  1+(readWidth / m_Param.m_StrideX);

    unsigned int filterHeight = filterShape[2];
    unsigned int readHeight = (inHeight + m_Param.m_PadTop + m_Param.m_PadBottom) - (filterHeight);
    unsigned int outHeight = 1+(readHeight / m_Param.m_StrideY);
    unsigned int depthMultiplier = filterShape[0];*/

    unsigned int outChannels = filterShape[1]*depthMultiplier;
    unsigned int outBatchSize = inBatchSize;

    TensorShape outShape({outBatchSize, outChannels, outHeight, outWidth});
    ConditionalThrow<LayerValidationException>(GetOutputSlot(0).ValidateTensorShape(outShape),
                                               "ReorgLayer: "
                                                       "TensorShape set on OutputSlot[0] does not match the inferred shape.");
}

}
