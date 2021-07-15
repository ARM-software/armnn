//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LayerWithParameters.hpp"

namespace armnn
{

class ScopedTensorHandle;

struct LstmOptLayerNormParameters
{
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_InputLayerNormWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_ForgetLayerNormWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_CellLayerNormWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_OutputLayerNormWeights;
};

struct LstmOptCifgParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_InputToInputWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_RecurrentToInputWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_InputGateBias;
};

struct LstmOptProjectionParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [output_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_ProjectionWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [output_size].
    std::shared_ptr<ConstTensorHandle> m_ProjectionBias;
};

struct LstmOptPeepholeParameters
{
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_CellToInputWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_CellToForgetWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_CellToOutputWeights;
};

struct LstmBasicParameters
{
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_InputToForgetWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_InputToCellWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [input_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_InputToOutputWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [output_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_RecurrentToForgetWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [output_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_RecurrentToCellWeights;
    /// A unique pointer to represent 2D weights tensor with dimensions [output_size, num_units].
    std::shared_ptr<ConstTensorHandle> m_RecurrentToOutputWeights;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_ForgetGateBias;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_CellBias;
    /// A unique pointer to represent 1D weights tensor with dimensions [num_units].
    std::shared_ptr<ConstTensorHandle> m_OutputGateBias;
};

} // namespace
