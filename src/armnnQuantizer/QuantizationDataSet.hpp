//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <map>
#include "QuantizationInput.hpp"
#include "armnn/LayerVisitorBase.hpp"
#include "armnn/Tensor.hpp"

namespace armnnQuantizer
{

/// QuantizationDataSet is a structure which is created after parsing a quantization CSV file.
/// It contains records of filenames which contain refinement data per pass ID for binding ID.
class QuantizationDataSet
{
    using QuantizationInputs = std::vector<armnnQuantizer::QuantizationInput>;
public:

    using iterator = QuantizationInputs::iterator;
    using const_iterator = QuantizationInputs::const_iterator;

    QuantizationDataSet();
    QuantizationDataSet(std::string csvFilePath);
    ~QuantizationDataSet();
    bool IsEmpty() const {return m_QuantizationInputs.empty();}

    iterator begin() { return m_QuantizationInputs.begin(); }
    iterator end() { return m_QuantizationInputs.end(); }
    const_iterator begin() const { return m_QuantizationInputs.begin(); }
    const_iterator end() const { return m_QuantizationInputs.end(); }
    const_iterator cbegin() const { return m_QuantizationInputs.cbegin(); }
    const_iterator cend() const { return m_QuantizationInputs.cend(); }

private:
    void ParseCsvFile();

    QuantizationInputs m_QuantizationInputs;
    std::string m_CsvFilePath;
};

/// Visitor class implementation to gather the TensorInfo for LayerBindingID for creation of ConstTensor for Refine.
class InputLayerVisitor : public armnn::LayerVisitorBase<armnn::VisitorNoThrowPolicy>
{
public:
    void VisitInputLayer(const armnn::IConnectableLayer *layer, armnn::LayerBindingId id, const char* name);
    armnn::TensorInfo GetTensorInfo(armnn::LayerBindingId);
private:
    std::map<armnn::LayerBindingId, armnn::TensorInfo> m_TensorInfos;
};

} // namespace armnnQuantizer