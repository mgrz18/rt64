//
// RT64
//

#include "rt64_interpreter.h"

#include <atomic>
#include <cassert>
#include <cstdio>

//#define DUMP_DISPLAY_LISTS

namespace RT64 {
    static FILE *displayListFp = nullptr;

    // Interpreter

    Interpreter::Interpreter() {
        state = nullptr;
        hleGBI = nullptr;
        extendedFunction = gbiManager.getExtendedFunction();
    }

    void Interpreter::setup(State *state) {
        this->state = state;
    }

    void Interpreter::loadUCodeGBI(uint32_t textAddress, uint32_t dataAddress, bool resetFromTask) {
        if (!resetFromTask) {
            state->flush();
        }

        const uint32_t AddressMask = 0xFFFFF8;
        const uint32_t maskedTextAddress = textAddress & AddressMask;
        const uint32_t maskedDataAddress = dataAddress & AddressMask;
        if ((UCode.textAddress != maskedTextAddress) || (UCode.dataAddress != maskedDataAddress)) {
            hleGBI = gbiManager.getGBIForUCode(state->RDRAM, maskedTextAddress, maskedDataAddress);
            fprintf(stderr, "[loadUCodeGBI] text=0x%08X data=0x%08X hleGBI=%p map[0x00]=%p map[0xFF]=%p map[0xC5]=%p\n",
                    maskedTextAddress, maskedDataAddress, (void*)hleGBI,
                    hleGBI ? (void*)hleGBI->map[0x00] : nullptr,
                    hleGBI ? (void*)hleGBI->map[0xFF] : nullptr,
                    hleGBI ? (void*)hleGBI->map[0xC5] : nullptr);
            if (hleGBI != nullptr) {
                state->rsp->setGBI(hleGBI);
            }

            UCode.textAddress = maskedTextAddress;
            UCode.dataAddress = maskedDataAddress;
        }

        if (hleGBI != nullptr) {
            GBIReset resetFunction = resetFromTask ? hleGBI->resetFromTask : hleGBI->resetFromLoad;
            if (resetFunction != nullptr) {
                resetFunction(state);
            }
        }
    }

    void Interpreter::processRDPLists(uint32_t dlStartAdddress, DisplayList *dlStart, DisplayList *dlEnd) {
        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        GBI *rdpGBI = state->rdp->gbi;
        constexpr unsigned int opCodeMask = 0x3F;

        // Run the command interpreter.
        assert(rdpGBI != nullptr);
        DisplayList *dl = dlStart;
        uint8_t opCode;
        GBIFunction func;
        uint32_t cmdLength;
        size_t pendingCommandRemainingBytes = state->rdp->pendingCommandRemainingBytes;

        if (dlStart >= dlEnd) {
            state->dlCpuProfiler.end();
            return;
        }

        if (pendingCommandRemainingBytes != 0) {
            // Copy the remaining command bytes from the current displaylist
            uint32_t toCopy = (uint32_t)std::min(pendingCommandRemainingBytes, (uintptr_t)dlEnd - (uintptr_t)dl);
            memcpy(state->rdp->pendingCommandBuffer.data() + state->rdp->pendingCommandCurrentBytes, dl, toCopy);

            // Modify start to skip the copied bytes
            dl = (DisplayList *)(toCopy + (uintptr_t)dl);

            // Check if we've copied all of the bytes of the command into the buffer
            if (pendingCommandRemainingBytes == toCopy) {
                // All bytes have been copied, so run the completed command
                DisplayList *pendingCommand = (DisplayList *)state->rdp->pendingCommandBuffer.data();
                opCode = (pendingCommand->w0 >> 24) & opCodeMask;
                func = rdpGBI->map[opCode];

                if (func != nullptr) {
                    func(state, &pendingCommand);
                }
                else {
                    RT64_LOG_PRINTF("DL Parser ran into an unknown RDP opCode: %u / 0x%X", opCode, opCode);
                }

                state->rdp->pendingCommandCurrentBytes = 0;
                state->rdp->pendingCommandRemainingBytes = 0;
            }
            // Not all of the bytes were copied, so adjust RDP state accordingly and exit.
            else {
                state->rdp->pendingCommandCurrentBytes += toCopy;
                state->rdp->pendingCommandRemainingBytes -= toCopy;
                state->dlCpuProfiler.end();
                return;
            }
        }

        // Create a dummy pointer and pass that, since displaylist pointer incrementing is handled differently in LLE.
        DisplayList *dummy;
        while ((dl != nullptr) && ((dlEnd == nullptr) || (dl < dlEnd))) {
            opCode = (dl->w0 >> 24) & opCodeMask;

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                dummy = dl;
                extendedFunction(state, &dl);
                cmdLength = 1;
            }
            else {
                func = rdpGBI->map[opCode];
                cmdLength = state->rdp->commandWordLengths[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                // Check if this command is unfinished and store the partial contents if so.
                if (dl + cmdLength > dlEnd) {
                    uint32_t toCopy = (uint32_t)((uintptr_t)dlEnd - (uintptr_t)dl);
                    memcpy(state->rdp->pendingCommandBuffer.data(), dl, toCopy);
                    state->rdp->pendingCommandCurrentBytes = toCopy;
                    state->rdp->pendingCommandRemainingBytes = cmdLength * sizeof(DisplayList) - toCopy;
                    break;
                }

                if (func != nullptr) {
                    dummy = dl;
                    func(state, &dummy);
                }
                // unknown RDP opcode — silent advance
            }

            if (dl != nullptr) {
                dl += cmdLength;
            }
        }

        state->dlCpuProfiler.end();
    }

    void Interpreter::processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart) {
        processDisplayLists(dlStartAdddress, dlStart, 0);
    }

    void Interpreter::processDisplayLists(uint32_t dlStartAdddress, DisplayList *dlStart, uint32_t dlEndBound) {
        assert(hleGBI != nullptr);

        state->dlCpuProfiler.start();

        // Update the state with the current display list address.
        state->displayListAddress = dlStartAdddress;
        state->displayListCounter++;

        // Check RDRAM if required.
        state->checkRDRAM();

        static int dl_session = 0;
        dl_session++;
        bool debug_this = false;  // was 'dl_session <= 3' — silenced

        // Run the command interpreter.
        DisplayList *dl = dlStart;
        DisplayList *dlEnd = (dlEndBound != 0) ? (DisplayList*)(state->RDRAM + dlEndBound) : nullptr;
        uint8_t opCode;
        GBIFunction func;
        uint32_t cmd_count = 0;
        constexpr uint32_t SAFETY_LIMIT = 20000;  // Typical N64 frame DL is <2000 cmds; raised to accommodate deep sub-DL traversal via lazy shadow
        uint8_t unknown_opcode_seen = 0;
        const uint8_t* rdram_base = state->RDRAM;
        const uint8_t* rdram_end = rdram_base + 0x800000;  // 8MB N64 RDRAM limit
        // Ring buffer of last N cmds for SAFETY_ABORT forensics.
        constexpr uint32_t RING_SIZE = 2000;
        // Track if walker has followed a G_DL to a region outside [dlStart, dlEnd).
        // Once true, we disable the dlEnd bound check because sub-DLs are legitimately
        // at different addresses. Reset per processDisplayLists call.
        bool walker_strayed = false;
        struct RingEntry { uint32_t cnt; uint8_t op; uint32_t w0; uint32_t w1; uint32_t dl_off; };
        RingEntry ring[RING_SIZE] = {};
        uint32_t ring_idx = 0;
        while (dl != nullptr) {
            // Bounds check: dl must be within RDRAM. If a runDl/loadUCode took us elsewhere
            // we'd get wild reads; abort before segfault.
            if ((const uint8_t*)dl < rdram_base || (const uint8_t*)dl >= rdram_end - sizeof(DisplayList)) {
                fprintf(stderr, "[DL %d] dl pointer 0x%p OUT OF RDRAM, aborting at cmd #%u last_cmd_count=%u\n",
                    dl_session, (void*)dl, cmd_count, cmd_count);
                break;
            }
            opCode = (dl->w0 >> 24);
            cmd_count++;

            // per-cmd logging for first few to diagnose walker path
            if (getenv("GE_LOG_WALK") != nullptr && cmd_count <= 40) {
                fprintf(stderr, "[walk #%u] op=0x%02X w0=0x%08X w1=0x%08X dl=%p\n",
                    cmd_count, opCode, dl->w0, dl->w1, (void*)dl);
            }
            // Record into ring buffer
            uint32_t dl_phys = uint32_t((const uint8_t*)dl - rdram_base);
            ring[ring_idx] = { cmd_count, opCode, dl->w0, dl->w1, dl_phys };
            ring_idx = (ring_idx + 1) % RING_SIZE;
            // Cycle detection: if the same dl address is hit more than N times in a row,
            // the walker is in a G_DL loop that never terminates. Abort so workload can
            // flush and present what's been rendered so far.
            static thread_local uint32_t last_dl_phys = 0;
            static thread_local uint32_t repeat_count = 0;
            constexpr uint32_t CYCLE_THRESHOLD = 5;
            if (dl_phys == last_dl_phys) {
                repeat_count++;
            } else {
                // Check if this address was visited recently many times.
                uint32_t visits_in_ring = 0;
                for (uint32_t i = 0; i < RING_SIZE; i++) {
                    if (ring[i].dl_off == dl_phys) visits_in_ring++;
                }
                if (visits_in_ring >= CYCLE_THRESHOLD) {
                    fprintf(stderr, "[DL %d] CYCLE DETECTED: addr 0x%08X visited %u times in last %u cmds — aborting at cmd %u\n",
                        dl_session, dl_phys, visits_in_ring, RING_SIZE, cmd_count);
                    break;
                }
                repeat_count = 0;
                last_dl_phys = dl_phys;
            }
            if (cmd_count > SAFETY_LIMIT) {
                fprintf(stderr, "[DL %d] SAFETY ABORT after %u commands, last opcode=0x%02X dl=%p\n",
                    dl_session, cmd_count, opCode, (void*)dl);
                // Dump the last RING_SIZE cmds in chronological order
                fprintf(stderr, "[DL %d] Last %u cmds before abort:\n", dl_session, RING_SIZE);
                for (uint32_t i = 0; i < RING_SIZE; i++) {
                    const RingEntry &e = ring[(ring_idx + i) % RING_SIZE];
                    if (e.cnt == 0) continue;
                    fprintf(stderr, "  #%u op=0x%02X w0=0x%08X w1=0x%08X phys=0x%08X\n",
                        e.cnt, e.op, e.w0, e.w1, e.dl_off);
                }
                break;
            }
            // Track if walker has strayed from main DL region.
            if (dl < dlStart || (dlEnd != nullptr && dl >= dlEnd)) {
                walker_strayed = true;
            }
            // dlEnd bound only fires if walker is still in main DL range (hasn't strayed).
            if (!walker_strayed && dlEnd != nullptr && dl >= dlEnd) {
                fprintf(stderr, "[DL %d] reached DL end bound after %u commands (dl=%p end=%p last_op=0x%02X)\n",
                    dl_session, cmd_count - 1, (void*)dl, (void*)dlEnd, opCode);
                break;
            }

            if ((extendedOpCode != 0) && (opCode == extendedOpCode)) {
                extendedFunction(state, &dl);
            }
            else {
                func = hleGBI->map[opCode];

#       ifdef DUMP_DISPLAY_LISTS
                RT64_LOG_PRINTF("0x%08X 0x%08X", dl->w0, dl->w1);
#       endif

                if (func != nullptr) {
                    func(state, &dl);
                }
                else {
                    // Count unknown opcodes so we can prioritize implementation.
                    static std::atomic<uint32_t> opcode_counts[256] = {};
                    static std::atomic<uint64_t> opcode_sample_w01[256][2] = {};
                    uint32_t c = opcode_counts[opCode].fetch_add(1);
                    if (c < 3) {
                        // Save a few (w0, w1) samples for the first three occurrences
                        opcode_sample_w01[opCode][0].store(dl->w0);
                        opcode_sample_w01[opCode][1].store(dl->w1);
                    }
                    // Dump a summary every 100 unknown cmds
                    static std::atomic<uint32_t> unk_total = 0;
                    uint32_t t = unk_total.fetch_add(1);
                    if (t < 5 || (t % 100) == 0) {
                        fprintf(stderr, "[UNKNOWN OPCODE SUMMARY] session DL=%d\n", dl_session);
                        for (int op = 0; op < 256; op++) {
                            uint32_t n = opcode_counts[op].load();
                            if (n > 0) {
                                uint64_t w0 = opcode_sample_w01[op][0].load();
                                uint64_t w1 = opcode_sample_w01[op][1].load();
                                fprintf(stderr, "  0x%02X count=%u sample w0=0x%08X w1=0x%08X\n", op, n, (uint32_t)w0, (uint32_t)w1);
                            }
                        }
                    }
                }
            }

            if (dl != nullptr) {
                dl++;
            }
        }

        if (debug_this) {
            fprintf(stderr, "[DL %d] processDisplayLists DONE total_cmds=%u\n", dl_session, cmd_count);
        }

        // Always call fullSync at end if under GE_DEEP_SHADOW — the walker may abort
        // via CYCLE or SAFETY before hitting a real G_RDPFULLSYNC. Without a terminal
        // fullSync the workload never commits and nothing renders.
        if (getenv("GE_DEEP_SHADOW") != nullptr) {
            state->fullSync();
        }
        else if (getenv("GE_FLUSH_AT_DL_END") != nullptr) {
            state->flush();
        }

        state->dlCpuProfiler.end();
    }
};
