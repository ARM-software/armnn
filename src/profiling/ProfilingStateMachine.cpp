//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ProfilingStateMachine.hpp"

#include <armnn/Exceptions.hpp>

#include <sstream>

namespace armnn
{

namespace profiling
{

namespace
{

void ThrowStateTransitionException(ProfilingState expectedState, ProfilingState newState)
{
    std::stringstream ss;
    ss << "Cannot transition from state [" << GetProfilingStateName(expectedState) << "] "
       << "to state [" << GetProfilingStateName(newState) << "]";
    throw armnn::RuntimeException(ss.str());
}

} // Anonymous namespace

ProfilingState ProfilingStateMachine::GetCurrentState() const
{
    return m_State.load();
}

void ProfilingStateMachine::TransitionToState(ProfilingState newState)
{
    ProfilingState currentState = m_State.load(std::memory_order::memory_order_relaxed);

    switch (newState)
    {
    case ProfilingState::Uninitialised:
        do
        {
            if (!IsOneOfStates(currentState, ProfilingState::Uninitialised))
            {
                ThrowStateTransitionException(currentState, newState);
            }
        }
        while (!m_State.compare_exchange_strong(currentState, newState, std::memory_order::memory_order_relaxed));
        break;
    case  ProfilingState::NotConnected:
        do
        {
            if (!IsOneOfStates(currentState, ProfilingState::Uninitialised, ProfilingState::NotConnected,
                               ProfilingState::Active, ProfilingState::WaitingForAck))
            {
                ThrowStateTransitionException(currentState, newState);
            }
        }
        while (!m_State.compare_exchange_strong(currentState, newState, std::memory_order::memory_order_relaxed));
        break;
    case ProfilingState::WaitingForAck:
        do
        {
            if (!IsOneOfStates(currentState, ProfilingState::NotConnected, ProfilingState::WaitingForAck))
            {
                ThrowStateTransitionException(currentState, newState);
            }
        }
        while (!m_State.compare_exchange_strong(currentState, newState, std::memory_order::memory_order_relaxed));
        break;
    case ProfilingState::Active:
        do
        {
            if (!IsOneOfStates(currentState, ProfilingState::WaitingForAck, ProfilingState::Active))
            {
                ThrowStateTransitionException(currentState, newState);
            }
        }
        while (!m_State.compare_exchange_strong(currentState, newState, std::memory_order::memory_order_relaxed));
        break;
    default:
        break;
    }
}

void ProfilingStateMachine::Reset()
{
    m_State.store(ProfilingState::Uninitialised);
}

} // namespace profiling

} // namespace armnn
