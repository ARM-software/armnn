//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

//#include "ArmnnDriver.hpp"
#include "Converter.hpp"

#include <armnn/ArmNN.hpp>

#include <nnapi/IPreparedModel.h>
#include <nnapi/Result.h>
#include <nnapi/TypeUtils.h>
#include <nnapi/Types.h>
#include <nnapi/Validation.h>

#include <set>
#include <map>
#include <vector>

namespace armnn_driver
{

using namespace android::nn;

// A helper template class performing the conversion from an AndroidNN driver Model representation,
// to an armnn::INetwork object
class ModelToINetworkTransformer
{
public:
    ModelToINetworkTransformer(const std::vector<armnn::BackendId>& backends,
                               const Model& model,
                               const std::set<unsigned int>& forcedUnsupportedOperations);

    ConversionResult GetConversionResult() const { return m_ConversionResult; }

    // Returns the ArmNN INetwork corresponding to the input model, if preparation went smoothly, nullptr otherwise.
    armnn::INetwork* GetINetwork() const { return m_Data.m_Network.get(); }

    bool IsOperationSupported(uint32_t operationIndex) const;

private:
    void Convert();

    // Shared aggregate input/output/internal data
    ConversionData m_Data;

    // Input data
    const Model&                  m_Model;
    const std::set<unsigned int>& m_ForcedUnsupportedOperations;

    // Output data
    ConversionResult         m_ConversionResult;
    std::map<uint32_t, bool> m_OperationSupported;
};

} // armnn_driver
