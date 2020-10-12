//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "SendThread.hpp"
#include "ProfilingUtils.hpp"

#include <armnn/Exceptions.hpp>
#include <armnn/Conversion.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <Processes.hpp>

#include <cstring>

namespace armnn
{

namespace profiling
{

SendThread::SendThread(armnn::profiling::ProfilingStateMachine& profilingStateMachine,
                       armnn::profiling::IBufferManager& buffer,
                       armnn::profiling::ISendCounterPacket& sendCounterPacket,
                       int timeout)
    : m_StateMachine(profilingStateMachine)
    , m_BufferManager(buffer)
    , m_SendCounterPacket(sendCounterPacket)
    , m_Timeout(timeout)
    , m_IsRunning(false)
    , m_KeepRunning(false)
    , m_SendThreadException(nullptr)
{
    m_BufferManager.SetConsumer(this);
}

void SendThread::SetReadyToRead()
{
    // We need to wait for the send thread to release its mutex
    {
        std::lock_guard<std::mutex> lck(m_WaitMutex);
        m_ReadyToRead = true;
    }
    // Signal the send thread that there's something to read in the buffer
    m_WaitCondition.notify_one();
}

void SendThread::Start(IProfilingConnection& profilingConnection)
{
    // Check if the send thread is already running
    if (m_IsRunning.load())
    {
        // The send thread is already running
        return;
    }

    if (m_SendThread.joinable())
    {
        m_SendThread.join();
    }

    // Mark the send thread as running
    m_IsRunning.store(true);

    // Keep the send procedure going until the send thread is signalled to stop
    m_KeepRunning.store(true);

    // Make sure the send thread will not flush the buffer until signaled to do so
    // no need for a mutex as the send thread can not be running at this point
    m_ReadyToRead = false;

    m_PacketSent = false;

    // Start the send thread
    m_SendThread = std::thread(&SendThread::Send, this, std::ref(profilingConnection));
}

void SendThread::Stop(bool rethrowSendThreadExceptions)
{
    // Signal the send thread to stop
    m_KeepRunning.store(false);

    // Check that the send thread is running
    if (m_SendThread.joinable())
    {
        // Kick the send thread out of the wait condition
        SetReadyToRead();
        // Wait for the send thread to complete operations
        m_SendThread.join();
    }

    // Check if the send thread exception has to be rethrown
    if (!rethrowSendThreadExceptions)
    {
        // No need to rethrow the send thread exception, return immediately
        return;
    }

    // Check if there's an exception to rethrow
    if (m_SendThreadException)
    {
        // Rethrow the send thread exception
        std::rethrow_exception(m_SendThreadException);

        // Nullify the exception as it has been rethrown
        m_SendThreadException = nullptr;
    }
}

void SendThread::Send(IProfilingConnection& profilingConnection)
{
    // Run once and keep the sending procedure looping until the thread is signalled to stop
    do
    {
        // Check the current state of the profiling service
        ProfilingState currentState = m_StateMachine.GetCurrentState();
        switch (currentState)
        {
        case ProfilingState::Uninitialised:
        case ProfilingState::NotConnected:

            // The send thread cannot be running when the profiling service is uninitialized or not connected,
            // stop the thread immediately
            m_KeepRunning.store(false);
            m_IsRunning.store(false);

            // An exception should be thrown here, save it to be rethrown later from the main thread so that
            // it can be caught by the consumer
            m_SendThreadException =
                    std::make_exception_ptr(RuntimeException("The send thread should not be running with the "
                                                             "profiling service not yet initialized or connected"));

            return;
        case ProfilingState::WaitingForAck:

            // Send out a StreamMetadata packet and wait for the profiling connection to be acknowledged.
            // When a ConnectionAcknowledged packet is received, the profiling service state will be automatically
            // updated by the command handler

            // Prepare a StreamMetadata packet and write it to the Counter Stream buffer
            m_SendCounterPacket.SendStreamMetaDataPacket();

             // Flush the buffer manually to send the packet
            FlushBuffer(profilingConnection);

            // Wait for a connection ack from the remote server. We should expect a response within timeout value.
            // If not, drop back to the start of the loop and detect somebody closing the thread. Then send the
            // StreamMetadata again.

            // Wait condition lock scope - Begin
            {
                std::unique_lock<std::mutex> lock(m_WaitMutex);

                bool timeout = m_WaitCondition.wait_for(lock,
                                                        std::chrono::milliseconds(std::max(m_Timeout, 1000)),
                                                        [&]{ return m_ReadyToRead; });
                // If we get notified we need to flush the buffer again
                if(timeout)
                {
                    // Otherwise if we just timed out don't flush the buffer
                    continue;
                }
                //reset condition variable predicate for next use
                m_ReadyToRead = false;
            }
            // Wait condition lock scope - End
            break;
        case ProfilingState::Active:
        default:
            // Wait condition lock scope - Begin
            {
                std::unique_lock<std::mutex> lock(m_WaitMutex);

                // Normal working state for the send thread
                // Check if the send thread is required to enforce a timeout wait policy
                if (m_Timeout < 0)
                {
                    // Wait indefinitely until notified that something to read has become available in the buffer
                    m_WaitCondition.wait(lock, [&] { return m_ReadyToRead; });
                }
                else
                {
                    // Wait until the thread is notified of something to read from the buffer,
                    // or check anyway after the specified number of milliseconds
                    m_WaitCondition.wait_for(lock, std::chrono::milliseconds(m_Timeout), [&] { return m_ReadyToRead; });
                }

                //reset condition variable predicate for next use
                m_ReadyToRead = false;
            }
            // Wait condition lock scope - End
            break;
        }

        // Send all the available packets in the buffer
        FlushBuffer(profilingConnection);
    } while (m_KeepRunning.load());

    // Ensure that all readable data got written to the profiling connection before the thread is stopped
    // (do not notify any watcher in this case, as this is just to wrap up things before shutting down the send thread)
    FlushBuffer(profilingConnection, false);

    // Mark the send thread as not running
    m_IsRunning.store(false);
}

void SendThread::FlushBuffer(IProfilingConnection& profilingConnection, bool notifyWatchers)
{
    // Get the first available readable buffer
    IPacketBufferPtr packetBuffer = m_BufferManager.GetReadableBuffer();

    // Initialize the flag that indicates whether at least a packet has been sent
    bool packetsSent = false;

    while (packetBuffer != nullptr)
    {
        // Get the data to send from the buffer
        const unsigned char* readBuffer = packetBuffer->GetReadableData();
        unsigned int readBufferSize = packetBuffer->GetSize();

        if (readBuffer == nullptr || readBufferSize == 0)
        {
            // Nothing to send, get the next available readable buffer and continue
            m_BufferManager.MarkRead(packetBuffer);
            packetBuffer = m_BufferManager.GetReadableBuffer();

            continue;
        }

        // Check that the profiling connection is open, silently drop the data and continue if it's closed
        if (profilingConnection.IsOpen())
        {
            // Write a packet to the profiling connection. Silently ignore any write error and continue
            profilingConnection.WritePacket(readBuffer, armnn::numeric_cast<uint32_t>(readBufferSize));

            // Set the flag that indicates whether at least a packet has been sent
            packetsSent = true;
        }

        // Mark the packet buffer as read
        m_BufferManager.MarkRead(packetBuffer);

        // Get the next available readable buffer
        packetBuffer = m_BufferManager.GetReadableBuffer();
    }
    // Check whether at least a packet has been sent
    if (packetsSent && notifyWatchers)
    {
        // Wait for the parent thread to release its mutex if necessary
        {
            std::lock_guard<std::mutex> lck(m_PacketSentWaitMutex);
            m_PacketSent = true;
        }
        // Notify to any watcher that something has been sent
        m_PacketSentWaitCondition.notify_one();
    }
}

bool SendThread::WaitForPacketSent(uint32_t timeout = 1000)
{
    std::unique_lock<std::mutex> lock(m_PacketSentWaitMutex);
    // Blocks until notified that at least a packet has been sent or until timeout expires.
    bool timedOut = m_PacketSentWaitCondition.wait_for(lock,
                                                       std::chrono::milliseconds(timeout),
                                                       [&] { return m_PacketSent; });

    m_PacketSent = false;

    return timedOut;
}

} // namespace profiling

} // namespace armnn
