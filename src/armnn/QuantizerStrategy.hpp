//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "Network.hpp"
#include "NetworkQuantizerUtils.hpp"
#include "StaticRangeStrategy.hpp"

#include <armnn/utility/NumericCast.hpp>
#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{
class QuantizerStrategy : public IStrategy
{
public :
    QuantizerStrategy(const RangeTracker& rangeTracker,
                      const IQuantizationScheme* quantizationScheme,
                      bool preserveType);

    ~QuantizerStrategy() = default;

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id) override;

    /// Extract the quantized network
    INetworkPtr RetrieveFinalNetwork() { return std::move(m_QuantizedNetwork); }

private:
    /// Connects the layer to preceeding layers and sets the quantization parameters based on recorded ranges
    void SetQuantizedInputConnections(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer);

    /// Record the guids so we can easily find the layers later
    void RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* qLayer);

    /// Sets the bias quantization scale based on input and weight scales
    ConstTensor CreateQuantizedBias(const IConnectableLayer* srcLayer,
                                    const ConstTensor& weights,
                                    const Optional<ConstTensor>& biases,
                                    std::vector<int32_t>& weightsBacking);

    /// Reference to the static range visitor used to retrieve the quantization ranges
    const RangeTracker& m_Ranges;

    /// Quantized version of the model we are building up
    INetworkPtr m_QuantizedNetwork;

    /// Mapping from input network guids to quantized network guids
    std::unordered_map<LayerGuid, LayerGuid> m_OriginalToQuantizedGuidMap;

    /// Mapping from guid to layer in quantized network
    std::unordered_map<LayerGuid, IConnectableLayer*> m_QuantizedGuidToLayerMap;

    const IQuantizationScheme* m_QuantizationScheme;

    const bool m_PreserveType;
};

} //namespace armnn