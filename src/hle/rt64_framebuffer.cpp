//
// RT64
//

#include "rt64_framebuffer.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <memory.h>
#include <stdlib.h>

#include "xxHash/xxh3.h"

#include "common/rt64_common.h"
#include "common/rt64_elapsed_timer.h"

#include "render/rt64_render_worker.h"

#ifdef __GNUC__
#define _byteswap_ulong __builtin_bswap32
#endif

#ifndef NDEBUG
//# define DUMP_RAW_RDRAM
#endif

namespace RT64 {
    // Framebuffer

    Framebuffer::Framebuffer() {
        addressStart = 0;
        addressEnd = 0;
        siz = 0;
        width = 0;
        height = 0;
        maxHeight = 0;
        modifiedBytes = 0;
        RAMBytes = 0;
        RAMHash = 0;
        ditherPatterns.fill(0);
        lastWriteType = Type::None;
        lastWriteFmt = 0;
        lastWriteTimestamp = 0;
        widthChanged = false;
        sizChanged = false;
        rdramChanged = false;
        interpolationEnabled = false;
        everUsedAsDepth = false;
    }

    Framebuffer::~Framebuffer() { }

    uint32_t Framebuffer::imageRowBytes(uint32_t rowWidth) const {
        return rowWidth << siz >> 1;
    }

    bool Framebuffer::contains(uint32_t start, uint32_t end) const {
        return (start >= addressStart) && (end <= addressEnd);
    }

    bool Framebuffer::overlaps(uint32_t start, uint32_t end) const {
        return (addressStart < end) && (addressEnd > start);
    }

    void Framebuffer::discardLastWrite() {
        lastWriteType = Type::None;
        lastWriteRect.reset();
    }

    bool Framebuffer::isLastWriteDifferent(Framebuffer::Type newType) const {
        return (lastWriteType != Type::None) && (lastWriteType != newType);
    }

    uint32_t Framebuffer::copyRAMToNativeAndChanges(RenderWorker *worker, FramebufferChange &fbChange, const uint8_t *src, uint32_t rowStart, uint32_t rowCount, uint8_t fmt, bool invalidateTargets, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);
        // Guard: if src is null (shouldn't happen but some code paths may pass null
        // when FB tracking gets confused by our shadow remap), skip the copy gracefully
        // instead of crashing in memcpy.
        if (src == nullptr) {
            static int null_log = 0;
            if (++null_log <= 5) {
                fprintf(stderr, "[Framebuffer::copyRAMToNativeAndChanges] SKIP: null src (fb width=%u rowCount=%u)\n", width, rowCount);
            }
            return 0;
        }
        // Sanity check: width must be reasonable (1-1024). Guards against corrupt FB
        // state when SETCIMG was misinterpreted.
        if (width == 0 || width > 1024 || rowCount == 0 || rowCount > 512) {
            static int bad_log = 0;
            if (++bad_log <= 5) {
                fprintf(stderr, "[Framebuffer::copyRAMToNativeAndChanges] SKIP: bad dims w=%u h=%u\n", width, rowCount);
            }
            return 0;
        }

        // Swap the endianness from the source.
        const uint32_t nativeSize = NativeTarget::getNativeSize(width, rowCount, siz);
        if (nativeSwappedRAM.size() < nativeSize) {
            nativeSwappedRAM.resize(nativeSize);
        }

        const uint32_t *srcWords = reinterpret_cast<const uint32_t *>(src);
        uint32_t *dstWords = reinterpret_cast<uint32_t *>(nativeSwappedRAM.data());
        uint32_t wordsToSwap = nativeSize / sizeof(uint32_t);
        while (wordsToSwap > 0) {
            *dstWords = _byteswap_ulong(*srcWords);
            wordsToSwap--;
            srcWords++;
            dstWords++;
        }

        uint32_t differentPixels = nativeTarget.copyFromRAM(worker, fbChange, width, rowCount, rowStart, siz, fmt, nativeSwappedRAM.data(), invalidateTargets, shaderLibrary);
        return differentPixels;
    }

    FramebufferChange *Framebuffer::readChangeFromBytes(RenderWorker *worker, FramebufferChangePool &fbChangePool, Type type, uint8_t fmt, const uint8_t *src, uint32_t rowStart, uint32_t rowCount, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);
        assert(src != nullptr);
        
        FramebufferChange &changeUsed = fbChangePool.use(worker, (type == Type::Depth) ? FramebufferChange::Type::Depth : FramebufferChange::Type::Color, width, rowCount, shaderLibrary->usesHDR);
        uint32_t readPixels = copyRAMToNativeAndChanges(worker, changeUsed, src, rowStart, rowCount, fmt, true, shaderLibrary);
        if (readPixels > 0) {
            return &changeUsed;
        }
        else {
            return nullptr;
        }
    }

    FramebufferChange *Framebuffer::readChangeFromStorage(RenderWorker *worker, const FramebufferStorage &fbStorage, FramebufferChangePool &fbChangePool,
        Type type, uint8_t fmt, uint32_t maxFbPairIndex, uint32_t rowStart, uint32_t rowCount, const ShaderLibrary *shaderLibrary)
    {
        assert(worker != nullptr);

        uint32_t rowBytes = imageRowBytes(width);
        const FramebufferStorage::Handle *storageHandle = fbStorage.get(maxFbPairIndex, addressStart + rowBytes * rowStart);
        if (storageHandle != nullptr) {
            const uint8_t *RDRAM = fbStorage.getRDRAM(*storageHandle);
            return readChangeFromBytes(worker, fbChangePool, type, fmt, RDRAM, rowStart, rowCount, shaderLibrary);
        }
        else {
            return nullptr;
        }
    }

    void Framebuffer::copyRenderTargetToNative(RenderWorker *worker, RenderTarget *target, uint32_t dstRowWidth, uint32_t dstRowStart, uint32_t dstRowEnd, uint8_t fmt, uint32_t ditherRandomSeed, const ShaderLibrary *shaderLibrary) {
        assert(worker != nullptr);
        static int rt_trace = 0;
        if (++rt_trace <= 15) {
            fprintf(stderr, "[copyRenderTargetToNative trace #%d] target=%p dstRowWidth=%u height=%u [%u..%u) addr=0x%08X\n",
                rt_trace, (void*)target, dstRowWidth, height, dstRowStart, dstRowEnd, addressStart);
        }
        // Guard against FBs created with garbage SETCIMG where dims are nonsensical,
        // or FBs at suspect addresses (near RDRAM end / outside sane range).
        if (target == nullptr || dstRowWidth == 0 || dstRowWidth > 1024 ||
            dstRowStart >= height || dstRowEnd > height || dstRowEnd <= dstRowStart ||
            addressStart >= 0x00800000 || width == 0 || width > 1024 || height == 0 || height > 512) {
            static int rt_skip_log = 0;
            if (++rt_skip_log <= 10) {
                fprintf(stderr, "[copyRenderTargetToNative] SKIP target=%p addr=0x%08X w=%u h=%u drw=%u [%u..%u)\n",
                    (void*)target, addressStart, width, height, dstRowWidth, dstRowStart, dstRowEnd);
            }
            return;
        }
        nativeTarget.copyToNative(worker, target, dstRowWidth, dstRowStart, dstRowEnd, siz, fmt, bestDitherPattern(), ditherRandomSeed, shaderLibrary);
    }

    // Track the most recently copied-back color FB address so the VI path can display
    // it even when the game's VI_ORIGIN_REG points at a different (stale) address.
    // Mutated only from gfx/framebuffer threads; read from VI thread. Plain atomics are
    // sufficient — occasional staleness is acceptable for display-only use.
    std::atomic<uint32_t> g_latestCopiedFbAddr{0};

    void Framebuffer::copyNativeToRAM(uint8_t *dst, uint32_t dstRowWidth, uint32_t dstRowStart, uint32_t dstRowEnd) {
        static int nt2ram_trace = 0;
        if (++nt2ram_trace <= 15) {
            fprintf(stderr, "[copyNativeToRAM TRACE #%d] addr=0x%08X w=%u h=%u drw=%u [%u..%u)\n",
                nt2ram_trace, addressStart, width, height, dstRowWidth, dstRowStart, dstRowEnd);
        }
        // Publish for the VI path (see rt64_render_context.cpp::update_screen under GE_FORCE_LATEST_FB).
        // Only publish plausible color FBs inside RDRAM.
        if (addressStart > 0 && addressStart < 0x00800000 && siz == 2) {
            g_latestCopiedFbAddr.store(addressStart, std::memory_order_relaxed);
        }
        if (dst == nullptr || dstRowWidth == 0 || dstRowWidth > 1024 ||
            dstRowStart >= height || dstRowEnd > height || dstRowEnd <= dstRowStart) {
            static int nt2ram_log = 0;
            if (++nt2ram_log <= 5) {
                fprintf(stderr, "[copyNativeToRAM] SKIP dst=%p w=%u h=%u [%u..%u)\n",
                    (void*)dst, dstRowWidth, height, dstRowStart, dstRowEnd);
            }
            return;
        }

        // Copy native target to RDRAM.
        uint8_t *dstBytes = dst + dstRowStart * imageRowBytes(dstRowWidth);
        uint32_t *dstWords = reinterpret_cast<uint32_t *>(dstBytes);
        uint32_t bytesToSwap = (dstRowEnd - dstRowStart) * imageRowBytes(dstRowWidth);
        uint32_t dstFirstWord = dstWords[0];
        nativeTarget.copyToRAM(dstRowStart, dstRowEnd, dstRowWidth, siz, dstBytes);

        // Write back to RDRAM by swapping every word.
        if (bytesToSwap >= sizeof(uint32_t)) {
            uint32_t wordsToSwap = (bytesToSwap) / sizeof(uint32_t);
            while (wordsToSwap > 0) {
                *dstWords = _byteswap_ulong(*dstWords);
                wordsToSwap--;
                dstWords++;
            }
        }
        // Special case when the total amount of bytes is smaller than a word.
        else {
            uint8_t *dstFirstWordU8 = reinterpret_cast<uint8_t *>(&dstFirstWord);
            for (uint32_t i = 0; i < bytesToSwap; i++) {
                dstFirstWordU8[i ^ 3] = dstBytes[i];
            }

            dstWords[0] = dstFirstWord;
        }

#   ifdef DUMP_RAW_RDRAM
        char path[1024];
        snprintf(path, sizeof(path), "dumps/FB_0x%X_W_%d_H_%d_SIZ_%d.rdram", addressStart, dstRowWidth, height, siz);
        FILE *fp = fopen(path, "wb");
        fwrite(dst, imageRowBytes(dstRowWidth) * dstRowEnd, 1, fp);
        fclose(fp);
#   endif
    }

    void Framebuffer::clearChanged() {
        widthChanged = false;
        sizChanged = false;
        rdramChanged = false;
    }

    void Framebuffer::addDitherPatterns(const std::array<uint32_t, 4> &extraPatterns) {
        for (uint32_t i = 0; i < ditherPatterns.size(); i++) {
            ditherPatterns[i] += extraPatterns[i];
        }
    }
    
    uint32_t Framebuffer::bestDitherPattern() const {
        return std::max_element(ditherPatterns.begin(), ditherPatterns.end()) - ditherPatterns.begin();
    }
};