//
// RT64
//

#include "rt64_rsp.h"

#include <cassert>
#include <cmath>

#include "../include/rt64_extended_gbi.h"
#include "common/rt64_common.h"
#include "common/rt64_math.h"
#include "gbi/rt64_f3d.h"
#include "shared/rt64_rsp_fog.h"

#include "rt64_interpreter.h"
#include "rt64_state.h"

//#define LOG_SPECIAL_MATRIX_OPERATIONS

// Forward decl of global shadow info struct defined in ultramodern/events.cpp.
struct GE_ShadowInfo {
    uint32_t src; uint32_t size; uint32_t dest; bool active;
};
extern GE_ShadowInfo g_ge_shadow;

// VI_ORIGIN_REG lives at global scope in rt64_render_context.cpp.
extern unsigned int VI_ORIGIN_REG;

namespace RT64 {
    // RSP

    constexpr float DepthRange = 1024.0f;

    RSP::RSP(State *state) {
        this->state = state;

        NoN = false;
        cullBothMask = 0;
        cullFrontMask = 0;
        projMask = 0;
        loadMask = 0;
        pushMask = 0;
        shadingSmoothMask = 0;

        segments.fill(0);
        reset();
    }

    void RSP::reset() {
        modelMatrixStackSize = 1;
        projectionMatrixStackSize = 1;
        viewportStackSize = 1;
        geometryModeStackSize = 1;
        otherModeStackSize = 1;
        // Initialize matrix stacks with IDENTITY (not zero). GE's ucode never explicitly
        // sets projection via a SETCIMG-style opcode; it relies on DMEM-resident matrices
        // pre-uploaded elsewhere. Initializing to zero caused vertex × modelview × ZERO = 0,
        // collapsing all transformed vertices to origin — invisible on screen.
        const hlslpp::float4x4 identity = hlslpp::float4x4(
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f);
        modelMatrixStack.fill(identity);
        modelMatrixSegmentedAddressStack.fill(0);
        modelMatrixPhysicalAddressStack.fill(0);
        viewMatrixStack[0] = identity;
        projMatrixStack[0] = identity;
        viewProjMatrixStack[0] = identity;
        invViewProjMatrixStack[0] = identity;
        vertices.fill({});
        indices.fill(0);
        used.reset();
        lights.fill({});
        segments.fill(0);
        viewportStack[0] = {};
        textureState = {};
        curViewProjIndex = 0;
        curTransformIndex = 0;
        curFogIndex = 0;
        curLightIndex = 0;
        curLightCount = 0;
        curLookAtIndex = 0;
        projectionIndex = -1;
        projectionMatrixChanged = false;
        projectionMatrixInversed = false;
        viewportChanged = false;
        modelViewProjMatrix = hlslpp::float4x4(0.0f);
        modelViewProjChanged = false;
        modelViewProjInserted = false;
        lightCount = 0;
        lightsChanged = false;
        vertexFogIndex = 0;
        vertexLightIndex = 0;
        vertexLightCount = 0;
        vertexLookAtIndex = 0;
        vertexColorPDAddress = 0;
        fog.mul = 0.0f;
        fog.offset = 0.0f;
        lookAt.x = { 0.0f, 0.0f, 0.0f };
        lookAt.y = { 0.0f, 0.0f, 0.0f };
        otherModeStack[0].L = 0x0;
        otherModeStack[0].H = 0x080CFF;
        geometryModeStack[0] = G_CLIPPING;
        fogChanged = true;
        lookAtChanged = true;

        clearExtended();
    }

    // Masks addresses as the RSP DMA hardware would.
    template<uint32_t mask> uint32_t RSP::maskPhysicalAddress(uint32_t address) {
        if (state->extended.extendRDRAM && ((address & 0xF0000000) == 0x80000000)) {
            return address - 0x80000000;
        }
        return address & mask;
    }

    // Performs a lookup in the segment table to convert the given address.
    uint32_t RSP::fromSegmented(uint32_t segAddress) {
        if (state->extended.extendRDRAM && ((segAddress & 0xF0000000) == 0x80000000)) {
            return segAddress;
        }
        return segments[((segAddress) >> 24) & 0x0F] + ((segAddress) & 0x00FFFFFF);
    }

    // Helper to remap a resolved phys address into shadow if it falls in shadow src range.
    static uint32_t ge_remap_to_shadow(uint32_t phys) {
        if (getenv("GE_DEEP_SHADOW") != nullptr && g_ge_shadow.active &&
            phys >= g_ge_shadow.src && phys < g_ge_shadow.src + g_ge_shadow.size) {
            return g_ge_shadow.dest + (phys - g_ge_shadow.src);
        }
        return phys;
    }

    // Converts the given segmented address and then applies the RSP DMA physical address mask.
    // Used in cases where the RSP performs a DMA with a segmented address as the input. 
    uint32_t RSP::fromSegmentedMasked(uint32_t segAddress) {
        return maskPhysicalAddress<0x00FFFFF8>(fromSegmented(segAddress));
    }

    uint32_t RSP::fromSegmentedMaskedPD(uint32_t segAddress) {
        return maskPhysicalAddress<0x00FFFFFC>(fromSegmented(segAddress));
    }

    void RSP::setSegment(uint32_t seg, uint32_t address) {
        assert(seg < RSP_MAX_SEGMENTS);
        uint32_t orig_addr = address;
        // If deep_shadow is active and segment base falls in shadowed range, remap to
        // shadow equivalent so all segmented-address G_DLs (0x02XXXXXX etc.) resolve
        // into the shadow copy instead of the game's heap (which may be overwritten).
        if (getenv("GE_DEEP_SHADOW") != nullptr && g_ge_shadow.active) {
            // Remap segments 2, 3, 6, 15 (observed DL-carrying segments). Segment 14
            // is used for RAM base (0x80000000) and should stay. Other segments untouched.
            bool is_dl_segment = (seg == 2 || seg == 3 || seg == 6 || seg == 15);
            if (is_dl_segment) {
                uint32_t phys = address & 0x00FFFFFF;
                if (phys >= g_ge_shadow.src && phys < g_ge_shadow.src + g_ge_shadow.size) {
                    address = g_ge_shadow.dest + (phys - g_ge_shadow.src);
                }
            }
        }
        static int seg_log = 0;
        if (++seg_log <= 20) {
            fprintf(stderr, "[RSP::setSegment #%d] seg=%u -> addr=0x%08X%s\n",
                seg_log, seg, address,
                (orig_addr != address) ? " (remapped to shadow)" : "");
        }
        segments[seg] = address;
    }

    void RSP::matrix(uint32_t address, uint8_t params) {
        uint32_t rdramAddress = fromSegmentedMasked(address);
        // 2026-05-04: do NOT shadow-remap matrices. The C-side allocates matrices
        // separately via osVirtualToPhysical() — they live OUTSIDE the DL stream.
        // The previous unconditional remap was reading matrix bytes from a stale
        // DL-stream snapshot, producing garbage values like (60, -28927, 0.003).
        // Gate via GE_SHADOW_MATRICES=1 if you ever need the old behavior back.
        if (getenv("GE_SHADOW_MATRICES") != nullptr) {
            rdramAddress = ge_remap_to_shadow(rdramAddress);
        }
        // Experiment: if GE_LOCK_MATRICES is set, only accept matrix loads from our
        // injection region (0x007F0000-0x007F01FF). Silently drop every other mtx cmd
        // so the game's own (broken) matrices don't overwrite our good ortho+identity.
        if (getenv("GE_LOCK_MATRICES") != nullptr &&
            (rdramAddress < 0x007F0000 || rdramAddress > 0x007F01FF)) {
            return;
        }
        const FixedMatrix *fixedMatrix = reinterpret_cast<FixedMatrix *>(state->fromRDRAM(rdramAddress));
        const hlslpp::float4x4 floatMatrix = fixedMatrix->toMatrix4x4();

        // 2026-05-04: Heuristic-reject matrices that don't look like valid affine
        // transforms or projections. GE's DL contains gSPMatrix-equivalent commands
        // whose w1 sometimes points to non-matrix data (DL chunks in g_GfxMemPos heap).
        // Reading 64 bytes of those as Mtx produces gibberish that explodes vertex
        // transforms (NaN/inf coords, billions of pixels off-screen).
        //
        // Valid matrix sanity: a libultra Mtx is either affine (last row [0,0,0,1])
        // or a projection (last col [0,0,0,1] in column-major hlslpp). If neither,
        // and any value is unreasonably large, drop the load.
        // Gate via GE_NO_MTX_FILTER=1 to disable.
        if (getenv("GE_NO_MTX_FILTER") == nullptr) {
            bool affine_row = (std::abs(floatMatrix[3][0]) < 0.5f) && (std::abs(floatMatrix[3][1]) < 0.5f) &&
                              (std::abs(floatMatrix[3][2]) < 0.5f) && (std::abs(floatMatrix[3][3] - 1.0f) < 0.5f);
            bool affine_col = (std::abs(floatMatrix[0][3]) < 0.5f) && (std::abs(floatMatrix[1][3]) < 0.5f) &&
                              (std::abs(floatMatrix[2][3]) < 0.5f) && (std::abs(floatMatrix[3][3] - 1.0f) < 0.5f);
            // 2026-05-04: a libultra perspective matrix has m[2][3] ≈ -1, m[3][3] = 0,
            // and |m[3][2]| > 0 (the -2nf/(n-f) term). Accept transposed layout too
            // since hlslpp may interpret the storage column-major.
            bool projection_a = (std::abs(floatMatrix[2][3] + 1.0f) < 0.25f) &&
                                (std::abs(floatMatrix[3][3]) < 0.25f) &&
                                (std::abs(floatMatrix[3][2]) > 0.001f);
            bool projection_b = (std::abs(floatMatrix[3][2] + 1.0f) < 0.25f) &&
                                (std::abs(floatMatrix[3][3]) < 0.25f) &&
                                (std::abs(floatMatrix[2][3]) > 0.001f);
            bool projection = projection_a || projection_b;
            float max_val = 0.0f;
            for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) {
                float v = std::abs((float)floatMatrix[i][j]);
                if (v > max_val) max_val = v;
            }
            if (!affine_row && !affine_col && !projection && max_val > 100.0f) {
                static int reject_log = 0;
                if (++reject_log <= 10) {
                    fprintf(stderr, "[RSP::matrix MALFORMED #%d] addr=0x%08X max=%.1f m[3]=[%.2f %.2f %.2f %.2f]\n",
                        reject_log, rdramAddress, (double)max_val,
                        (double)floatMatrix[3][0], (double)floatMatrix[3][1],
                        (double)floatMatrix[3][2], (double)floatMatrix[3][3]);
                }
                // Diagnostic only: log malformed matrices but don't drop or substitute
                // (both options tested 2026-05-04 — drop produced NaN tris, substitute-
                // identity produced huge off-screen coords). Detection is reliable but
                // the right action depends on understanding what the game intends with
                // these non-Mtx-format references.
            }
        }
        // Debug dump: first few matrix loads with decoded values
        static int mtx_log = 0;
        if (++mtx_log <= 8) {
            fprintf(stderr, "[RSP::matrix #%d] seg=0x%08X phys=0x%08X params=0x%02X\n",
                mtx_log, address, rdramAddress, params);
            fprintf(stderr, "  decoded: [%f %f %f %f / %f %f %f %f / %f %f %f %f / %f %f %f %f]\n",
                (double)floatMatrix[0][0], (double)floatMatrix[0][1], (double)floatMatrix[0][2], (double)floatMatrix[0][3],
                (double)floatMatrix[1][0], (double)floatMatrix[1][1], (double)floatMatrix[1][2], (double)floatMatrix[1][3],
                (double)floatMatrix[2][0], (double)floatMatrix[2][1], (double)floatMatrix[2][2], (double)floatMatrix[2][3],
                (double)floatMatrix[3][0], (double)floatMatrix[3][1], (double)floatMatrix[3][2], (double)floatMatrix[3][3]);

        }

        // Projection matrix.
        hlslpp::float4x4 &viewMatrix = viewMatrixStack[projectionMatrixStackSize - 1];
        hlslpp::float4x4 &projMatrix = projMatrixStack[projectionMatrixStackSize - 1];
        hlslpp::float4x4 &viewProjMatrix = viewProjMatrixStack[projectionMatrixStackSize - 1];
        uint32_t &projectionMatrixSegmentedAddress = projectionMatrixSegmentedAddressStack[projectionMatrixStackSize - 1];
        uint32_t &projectionMatrixPhysicalAddress = projectionMatrixPhysicalAddressStack[projectionMatrixStackSize - 1];
        if (params & projMask) {
            if (params & loadMask) {
                viewProjMatrix = floatMatrix;

                if (isMatrixViewProj(floatMatrix)) {
                    matrixDecomposeViewProj(floatMatrix, viewMatrix, projMatrix);
                }
                else {
                    projMatrix = floatMatrix;
                    viewMatrix = hlslpp::float4x4::identity();
                }
            }
            else {
                viewProjMatrix = hlslpp::mul(floatMatrix, viewProjMatrix);

                if (isMatrixAffine(floatMatrix) && !isMatrixIdentity(floatMatrix)) {
                    viewMatrix = hlslpp::mul(floatMatrix, viewMatrix);
                }
                else {
                    projMatrix = hlslpp::mul(floatMatrix, projMatrix);
                }
            }

            projectionMatrixSegmentedAddress = address;
            projectionMatrixPhysicalAddress = rdramAddress;
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }
        // Modelview matrix.
        else {
            if ((params & pushMask) && (modelMatrixStackSize < RSP_MATRIX_STACK_SIZE)) {
                modelMatrixStackSize++;
                modelMatrixStack[modelMatrixStackSize - 1] = modelMatrixStack[modelMatrixStackSize - 2];
            }

            if (params & loadMask) {
                modelMatrixStack[modelMatrixStackSize - 1] = floatMatrix;
            }
            else {
                modelMatrixStack[modelMatrixStackSize - 1] = hlslpp::mul(floatMatrix, modelMatrixStack[modelMatrixStackSize - 1]);
            }

            modelMatrixSegmentedAddressStack[modelMatrixStackSize - 1] = address;
            modelMatrixPhysicalAddressStack[modelMatrixStackSize - 1] = rdramAddress;
        }
        
        modelViewProjChanged = true;
    }

    void RSP::popMatrix(uint32_t count) {
        while (count--) {
            if (modelMatrixStackSize > 1) {
                modelMatrixStackSize--;
                modelViewProjChanged = true;
            }
        }
    }

    void RSP::pushProjectionMatrix() {
        if (projectionMatrixStackSize < RSP_EXTENDED_STACK_SIZE) {
            viewMatrixStack[projectionMatrixStackSize] = viewMatrixStack[projectionMatrixStackSize - 1];
            projMatrixStack[projectionMatrixStackSize] = projMatrixStack[projectionMatrixStackSize - 1];
            viewProjMatrixStack[projectionMatrixStackSize] = viewProjMatrixStack[projectionMatrixStackSize - 1];
            invViewProjMatrixStack[projectionMatrixStackSize] = invViewProjMatrixStack[projectionMatrixStackSize - 1];
            projectionMatrixSegmentedAddressStack[projectionMatrixStackSize] = projectionMatrixSegmentedAddressStack[projectionMatrixStackSize - 1];
            projectionMatrixPhysicalAddressStack[projectionMatrixStackSize] = projectionMatrixPhysicalAddressStack[projectionMatrixStackSize - 1];
            projectionMatrixStackSize++;
        }
    }

    void RSP::popProjectionMatrix() {
        if (projectionMatrixStackSize > 1) {
            projectionMatrixStackSize--;
            modelViewProjChanged = true;
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }
    }
    
    void RSP::insertMatrix(uint32_t address, uint32_t value) {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::insertMatrix(dst %u, value %u)", dst, value);
#   endif

        // We assume unaligned addresses or overlapping is impossible until we have verification
        // from the microcode itself if this is allowed or not due to how much more complex
        // the implementation could get.
        assert(((address & 0x3) == 0) && "Unaligned addresses for insert matrix are not currently supported.");

        // Copies a 32-bit value into the location occupied by the fixed point matrices.
        const uint16_t MatrixSize = 0x40;
        const uint16_t FractionalMatrixAddress = MatrixSize / 2;
        const uint16_t ModelAddress = 0x0;
        const uint16_t ViewProjAddress = ModelAddress + MatrixSize;
        const uint16_t ModelViewProjAddress = ViewProjAddress + MatrixSize;

        // According to the microcode, the address requires this kind of wrapping
        // to access the real destination.
        uint32_t dstAddr = (address + ModelViewProjAddress) & 0xFFFFU;

        // Figure out which matrix should be modified and compute a relative address to it.
        uint32_t relAddr = 0;
        hlslpp::float4x4 *dstMat = nullptr;
        hlslpp::float4x4 &viewProjMatrix = viewProjMatrixStack[projectionMatrixStackSize - 1];
        if (dstAddr >= (ModelViewProjAddress + MatrixSize)) {
            assert(false && "Undefined behavior due to destination address extending outside of the allowed bounds.");
            return;
        }
        if (dstAddr >= ModelViewProjAddress) {
            dstMat = &modelViewProjMatrix;
            relAddr = dstAddr - ModelViewProjAddress;
            modelViewProjInserted = true;
        }
        else if (dstAddr >= ViewProjAddress) {
            dstMat = &viewProjMatrix;
            relAddr = dstAddr - ViewProjAddress;
            projectionMatrixChanged = true;
            projectionMatrixInversed = false;
        }
        else if (dstAddr >= ModelAddress) {
            dstMat = &modelMatrixStack[modelMatrixStackSize - 1];
            relAddr = dstAddr - ModelAddress;
        }

        // Modify two fractional parts or two integer parts.
        const bool modifyFractional = relAddr >= FractionalMatrixAddress;
        if (modifyFractional) {
            relAddr -= FractionalMatrixAddress;
        }

        const uint32_t index = relAddr / 2;
        const uint32_t row = index / 4;
        const uint32_t column = index % 4;
        if (modifyFractional) {
            FixedMatrix::modifyMatrix4x4Fraction(*dstMat, row, column, uint16_t((value >> 16U) & 0xFFFFU));
            FixedMatrix::modifyMatrix4x4Fraction(*dstMat, row, column + 1, uint16_t(value & 0xFFFFU));
        }
        else {
            FixedMatrix::modifyMatrix4x4Integer(*dstMat, row, column, int16_t((value >> 16) & 0xFFFF));
            FixedMatrix::modifyMatrix4x4Integer(*dstMat, row, column + 1, int16_t(value & 0xFFFF));
        }
    }

    void RSP::forceMatrix(uint32_t address) {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::forceMatrix(0x%08X)", address);
#   endif

        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const FixedMatrix *fixedMatrix = reinterpret_cast<FixedMatrix *>(state->fromRDRAM(rdramAddress));
        modelViewProjMatrix = fixedMatrix->toMatrix4x4();
        modelViewProjInserted = true;
        modelViewProjChanged = false;
    }

    void RSP::computeModelViewProj() {
        const hlslpp::float4x4 &viewProjMatrix = viewProjMatrixStack[projectionMatrixStackSize - 1];
        modelViewProjMatrix = hlslpp::mul(modelMatrixStack[modelMatrixStackSize - 1], viewProjMatrix);
        modelViewProjInserted = false;
        modelViewProjChanged = false;
    }

    void RSP::specialComputeModelViewProj() {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::specialComputeModelViewProj()");
#   endif

        computeModelViewProj();
    }

    void RSP::setModelViewProjChanged(bool changed) {
#   ifdef LOG_SPECIAL_MATRIX_OPERATIONS
        RT64_LOG_PRINTF("RSP::setModelViewProjChanged(%d)", changed);
#   endif

        modelViewProjChanged = changed;
    }

    void RSP::setVertex(uint32_t address, uint8_t vtxCount, uint32_t dstIndex) {
        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "Vertex indices are not valid. DL is possibly corrupted.");
            return;
        }

        uint32_t rdramAddress = fromSegmentedMasked(address);
        // Optional shadow remap for vertex data. Controlled by GE_REMAP_VTX env var
        // because vertex data may live in a different region than matrix/texture data.
        if (getenv("GE_REMAP_VTX") != nullptr) {
            rdramAddress = ge_remap_to_shadow(rdramAddress);
        }
        const Vertex *dlVerts = reinterpret_cast<const Vertex *>(state->fromRDRAM(rdramAddress));
        memcpy(&vertices[dstIndex], dlVerts, sizeof(Vertex) * vtxCount);
        // Debug: dump first few vertex loads to see coord range
        static int vtx_log = 0;
        if (++vtx_log <= 6) {
            fprintf(stderr, "[RSP::setVertex #%d] seg=0x%08X phys=0x%08X n=%u dst=%u\n",
                vtx_log, address, rdramAddress, vtxCount, dstIndex);
            for (uint32_t i = 0; i < vtxCount && i < 4; ++i) {
                const Vertex &v = vertices[dstIndex + i];
                fprintf(stderr, "  v[%u]: pos=(%d,%d,%d) st=(%d,%d) rgba=(%u,%u,%u,%u)\n",
                    i, v.x, v.y, v.z, v.s, v.t, v.color.r, v.color.g, v.color.b, v.color.a);
            }
        }
        setVertexCommon<true>(dstIndex, dstIndex + vtxCount);
    }
    
    void RSP::setVertexPD(uint32_t address, uint8_t vtxCount, uint32_t dstIndex) {
        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "Vertex indices are not valid. DL is possibly corrupted.");
            return;
        }

        const uint32_t rdramAddress = fromSegmentedMaskedPD(address);
        const VertexPD *dlVerts = reinterpret_cast<const VertexPD *>(state->fromRDRAM(rdramAddress));
        for (uint32_t i = 0; i < vtxCount; i++) {
            Vertex &dst = vertices[dstIndex + i];
            const VertexPD &src = dlVerts[i];
            const uint8_t *col = state->fromRDRAM(vertexColorPDAddress + (src.ci & 0xFFU));
            dst.x = src.x;
            dst.y = src.y;
            dst.z = src.z;
            dst.s = src.s;
            dst.t = src.t;
            dst.color.r = col[3];
            dst.color.g = col[2];
            dst.color.b = col[1];
            dst.color.a = col[0];
        }

        setVertexCommon<true>(dstIndex, dstIndex + vtxCount);
    }

    void RSP::setVertexEXV1(uint32_t address, uint8_t vtxCount, uint32_t dstIndex) {
        if ((dstIndex >= RSP_MAX_VERTICES) || ((dstIndex + vtxCount) > RSP_MAX_VERTICES)) {
            assert(false && "Vertex indices are not valid. DL is possibly corrupted.");
            return;
        }

        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const VertexEXV1 *dlVerts = reinterpret_cast<const VertexEXV1 *>(state->fromRDRAM(rdramAddress));
        auto &velShorts = workload.drawData.velShorts;
        for (uint32_t i = 0; i < vtxCount; i++) {
            const VertexEXV1 &src = dlVerts[i];
            velShorts.emplace_back(src.v.x - src.xp);
            velShorts.emplace_back(src.v.y - src.yp);
            velShorts.emplace_back(src.v.z - src.zp);
            vertices[dstIndex + i] = src.v;
        }

        setVertexCommon<false>(dstIndex, dstIndex + vtxCount);
    }

    void RSP::setVertexColorPD(uint32_t address) {
        vertexColorPDAddress = fromSegmentedMasked(address);
    }

    Projection::Type RSP::getCurrentProjectionType() const {
        const hlslpp::float4x4 &projMatrix = projMatrixStack[projectionMatrixStackSize - 1];
        const bool perspProj = (projMatrix[3][3] == 0.0f) && (abs(projMatrix[1][1]) > 1e-6f);
        if (perspProj) {
            return Projection::Type::Perspective;
        }
        else {
            return Projection::Type::Orthographic;
        }
    }
    
    void RSP::addCurrentProjection(Projection::Type type) {
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        if (extended.viewProjMatrixIdStackChanged) {
            extended.curViewProjMatrixIdGroupIndex = int(workload.drawData.transformGroups.size());
            workload.drawData.transformGroups.emplace_back(extended.viewProjMatrixIdStack[extended.viewProjMatrixIdStackSize - 1]);
            extended.viewProjMatrixIdStackChanged = false;
        }

        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        if (projectionMatrixChanged || viewportChanged) {
            DrawData &drawData = workload.drawData;
            uint32_t transformsIndex = uint32_t(drawData.viewTransforms.size());
            curViewProjIndex = transformsIndex;

            uint32_t physicalAddress = projectionMatrixPhysicalAddressStack[projectionMatrixStackSize - 1];
            workload.physicalAddressTransformMap.emplace(physicalAddress, uint32_t(drawData.viewProjTransformGroups.size()));
            drawData.viewTransforms.emplace_back(viewMatrixStack[projectionMatrixStackSize - 1]);
            drawData.projTransforms.emplace_back(projMatrixStack[projectionMatrixStackSize - 1]);
            drawData.viewProjTransforms.emplace_back(viewProjMatrixStack[projectionMatrixStackSize - 1]);
            drawData.viewProjTransformGroups.emplace_back(extended.curViewProjMatrixIdGroupIndex);
            drawData.rspViewports.emplace_back(viewportStack[viewportStackSize - 1]);
            drawData.viewportOrigins.emplace_back(extended.viewportOrigin);
            projectionMatrixChanged = false;
            viewportChanged = false;
        }

        projectionIndex = fbPair.changeProjection(curViewProjIndex, type);
    }

    template<bool addEmptyVelocity>
    void RSP::setVertexCommon(uint8_t dstIndex, uint8_t dstMax) {
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];

        if (extended.modelMatrixIdStackChanged) {
            const int stackIndex = extended.modelMatrixIdStackSize - 1;
            extended.curModelMatrixIdGroupIndex = int(workload.drawData.transformGroups.size());
            workload.drawData.transformGroups.emplace_back(extended.modelMatrixIdStack[stackIndex]);
            extended.modelMatrixIdStackChanged = false;
        }
        
        // ModelViewProj is only updated when a vertex is processed and the flag is enabled.
        auto &worldTransforms = workload.drawData.worldTransforms;
        auto &worldTransformGroups = workload.drawData.worldTransformGroups;
        auto &worldTransformSegmentedAddresses = workload.drawData.worldTransformSegmentedAddresses;
        auto &worldTransformPhysicalAddresses = workload.drawData.worldTransformPhysicalAddresses;
        auto &worldTransformVertexIndices = workload.drawData.worldTransformVertexIndices;
        bool addWorldTransform = (modelViewProjChanged || modelViewProjInserted);
        if (modelViewProjChanged) {
            computeModelViewProj();
            curTransformIndex = static_cast<uint16_t>(worldTransforms.size());
            worldTransforms.emplace_back(modelMatrixStack[modelMatrixStackSize - 1]);
        }
        else if (modelViewProjInserted) {
#       ifdef LOG_SPECIAL_MATRIX_OPERATIONS
            RT64_LOG_PRINTF("RSP::setVertexCommon(dstIndex %u, dstMax %u): An MVP was inserted and must be inversed.", dstIndex, dstMax);
#       endif

            modelViewProjInserted = false;

            if (!projectionMatrixInversed) {
                invViewProjMatrixStack[projectionMatrixStackSize - 1] = hlslpp::inverse(viewProjMatrixStack[projectionMatrixStackSize - 1]);
                projectionMatrixInversed = true;
            }

            curTransformIndex = static_cast<uint16_t>(worldTransforms.size());
            worldTransforms.emplace_back(hlslpp::mul(modelViewProjMatrix, invViewProjMatrixStack[projectionMatrixStackSize - 1]));
        }

        if (addWorldTransform) {
            uint32_t physicalAddress = modelMatrixPhysicalAddressStack[modelMatrixStackSize - 1];
            workload.physicalAddressTransformMap.emplace(physicalAddress, uint32_t(worldTransformGroups.size()));
            worldTransformGroups.emplace_back(extended.curModelMatrixIdGroupIndex);
            worldTransformSegmentedAddresses.emplace_back(modelMatrixSegmentedAddressStack[modelMatrixStackSize - 1]);
            worldTransformPhysicalAddresses.emplace_back(physicalAddress);
            worldTransformVertexIndices.emplace_back(workload.drawData.vertexCount());
        }

        // Push a new projection if it's changed.
        if ((projectionIndex < 0) || projectionMatrixChanged || viewportChanged) {
            state->flush();
            addCurrentProjection(getCurrentProjectionType());
        }

        // We push a new set of lights if a vertex actually uses it.
        const GBI *curGBI = state->ext.interpreter->hleGBI;
        uint32_t &geometryMode = geometryModeStack[geometryModeStackSize - 1];
        const bool usesLighting = (geometryMode & G_LIGHTING);
        const bool usesPointLighting = curGBI->flags.pointLighting && (geometryMode & G_POINT_LIGHTING);
        if (usesLighting) {
            if (lightsChanged) {
                auto &rspLights = workload.drawData.rspLights;
                vertexLightIndex = static_cast<uint32_t>(rspLights.size());
                vertexLightCount = lightCount + 1;
                for (int l = 0; l < lightCount + 1; l++) {
                    const Light &light = lights[l];

                    // Decode light into easier to use parameters.
                    interop::RSPLight rspLight;
                    rspLight.col = {
                        light.dir.colr / 255.0f,
                        light.dir.colg / 255.0f,
                        light.dir.colb / 255.0f
                    };

                    rspLight.colc = {
                        light.dir.colcr / 255.0f,
                        light.dir.colcg / 255.0f,
                        light.dir.colcb / 255.0f
                    };

                    if (usesPointLighting && (light.pos.kc > 0)) {
                        rspLight.posDir = {
                            static_cast<float>(light.pos.posx),
                            static_cast<float>(light.pos.posy),
                            static_cast<float>(light.pos.posz)
                        };

                        rspLight.kc = light.pos.kc;
                        rspLight.kl = light.pos.kl;
                        rspLight.kq = light.pos.kq;
                    }
                    else {
                        rspLight.posDir = {
                            static_cast<float>(light.dir.dirx),
                            static_cast<float>(light.dir.diry),
                            static_cast<float>(light.dir.dirz)
                        };

                        rspLight.kc = 0;
                        rspLight.kl = 0;
                        rspLight.kq = 0;
                    }

                    rspLights.push_back(rspLight);
                }

                state->updateDrawStatusAttribute(DrawAttribute::Lights);
                lightsChanged = false;
            }

            curLightIndex = vertexLightIndex;
            curLightCount = vertexLightCount;
        }
        else {
            curLightIndex = 0;
            curLightCount = 0;
        }

        const bool usesFog = (geometryMode & G_FOG);
        if (usesFog) {
            if (fogChanged) {
                auto &rspFogVector = workload.drawData.rspFog;
                vertexFogIndex = static_cast<uint32_t>(rspFogVector.size());
                rspFogVector.emplace_back(fog);
                fogChanged = false;
            }

            // Fog index is encoded as the real index + 1 to use 0 as the "fog disabled" index.
            curFogIndex = vertexFogIndex + 1;
        }
        else {
            curFogIndex = 0;
        }

        const uint32_t textureGenMask = G_LIGHTING | G_TEXTURE_GEN;
        const bool usesTextureGen = (geometryMode & textureGenMask) == textureGenMask;
        if (usesTextureGen) {
            if (lookAtChanged) {
                auto &rspLookAtVector = workload.drawData.rspLookAt;
                vertexLookAtIndex = static_cast<uint32_t>(rspLookAtVector.size());
                rspLookAtVector.emplace_back(lookAt);
                lookAtChanged = false;
            }

            // Look at index is encoded with one bit to determine if texture gen is enabled, another bit 
            // to determine if it's linear texture gen or not, and the rest of the bits to hold the real index.
            curLookAtIndex = RSP_LOOKAT_INDEX_ENABLED;
            curLookAtIndex |= (geometryMode & G_TEXTURE_GEN_LINEAR) ? RSP_LOOKAT_INDEX_LINEAR : 0x0;
            curLookAtIndex |= (vertexLookAtIndex << RSP_LOOKAT_INDEX_SHIFT);
        }
        else {
            curLookAtIndex = 0;
        }

        auto &posShorts = workload.drawData.posShorts;
        auto &velShorts = workload.drawData.velShorts;
        auto &tcFloats = workload.drawData.tcFloats;
        auto &normColBytes = workload.drawData.normColBytes;
        auto &viewProjIndices = workload.drawData.viewProjIndices;
        auto &worldIndices = workload.drawData.worldIndices;
        auto &fogIndices = workload.drawData.fogIndices;
        auto &lightIndices = workload.drawData.lightIndices;
        auto &lightCounts = workload.drawData.lightCounts;
        auto &lookAtIndices = workload.drawData.lookAtIndices;
        auto &posTransformed = workload.drawData.posTransformed;
        auto &posScreen = workload.drawData.posScreen;
        const auto &mvp = modelViewProjMatrix;
        const uint32_t globalIndex = workload.drawData.vertexCount();
        for (uint32_t i = dstIndex; i < dstMax; i++) {
            auto &v = vertices[i];
            posShorts.emplace_back(v.x);
            posShorts.emplace_back(v.y);
            posShorts.emplace_back(v.z);
            normColBytes.emplace_back(v.color.r);
            normColBytes.emplace_back(v.color.g);
            normColBytes.emplace_back(v.color.b);
            normColBytes.emplace_back(v.color.a);
            viewProjIndices.emplace_back(curViewProjIndex);
            worldIndices.emplace_back(curTransformIndex);
            fogIndices.emplace_back(curFogIndex);
            lightIndices.emplace_back(curLightIndex);
            lightCounts.emplace_back(curLightCount);
            lookAtIndices.emplace_back(curLookAtIndex);
            indices[i] = uint32_t(globalIndex) + (i - dstIndex);
            used[i] = false;

            if constexpr (addEmptyVelocity) {
                velShorts.emplace_back(0);
                velShorts.emplace_back(0);
                velShorts.emplace_back(0);
            }
        }

        static int transform_log = 0;
        if (++transform_log <= 3) {
            fprintf(stderr, "[setVertexCommon xform #%d] MVP=[%.6f %.6f %.6f %.6f / %.6f %.6f %.6f %.6f / %.6f %.6f %.6f %.6f / %.6f %.6f %.6f %.6f] vp.scale=(%.2f,%.2f,%.2f) vp.trans=(%.2f,%.2f,%.2f)\n",
                transform_log,
                (double)mvp[0][0], (double)mvp[0][1], (double)mvp[0][2], (double)mvp[0][3],
                (double)mvp[1][0], (double)mvp[1][1], (double)mvp[1][2], (double)mvp[1][3],
                (double)mvp[2][0], (double)mvp[2][1], (double)mvp[2][2], (double)mvp[2][3],
                (double)mvp[3][0], (double)mvp[3][1], (double)mvp[3][2], (double)mvp[3][3],
                (double)viewportStack[viewportStackSize-1].scale.x, (double)viewportStack[viewportStackSize-1].scale.y, (double)viewportStack[viewportStackSize-1].scale.z,
                (double)viewportStack[viewportStackSize-1].translate.x, (double)viewportStack[viewportStackSize-1].translate.y, (double)viewportStack[viewportStackSize-1].translate.z);
        }
        for (uint32_t i = dstIndex; i < dstMax; i++) {
            auto &v = vertices[i];
            const hlslpp::float4 tfPos = hlslpp::mul(hlslpp::float4(v.x, v.y, v.z, 1.0f), mvp);
            const interop::RSPViewport &viewport = viewportStack[viewportStackSize - 1];
            posTransformed.emplace_back(tfPos);
            posScreen.emplace_back((tfPos.xyz / hlslpp::float3(tfPos.w, -tfPos.w, tfPos.w)) * viewport.scale + viewport.translate);
        }

        if (usesTextureGen) {
            const float TextureSc = static_cast<float>(textureState.sc);
            const float TextureTc = static_cast<float>(textureState.tc);
            for (uint32_t i = dstIndex; i < dstMax; i++) {
                tcFloats.emplace_back(TextureSc);
                tcFloats.emplace_back(TextureTc);
            }
        }
        else {
            const int32_t TextureSc = (int32_t)(textureState.sc);
            const int32_t TextureTc = (int32_t)(textureState.tc);
            const double Divisor = 65536.0f * 32.0f;
            for (uint32_t i = dstIndex; i < dstMax; i++) {
                tcFloats.emplace_back((float)((double)((vertices[i].s) * TextureSc) / Divisor));
                tcFloats.emplace_back((float)((double)((vertices[i].t) * TextureTc) / Divisor));
            }
        }
    }

    void RSP::modifyVertex(uint16_t dstIndex, uint16_t dstAttribute, uint32_t value) {
        if (dstIndex >= RSP_MAX_VERTICES) {
            assert(false && "Vertex index is not valid. DL is possibly corrupted.");
            return;
        }

        // If the vertex was used already in the frame, then we create a new copy instead.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        auto &normColBytes = workload.drawData.normColBytes;
        auto &tcFloats = workload.drawData.tcFloats;
        auto &fogIndices = workload.drawData.fogIndices;
        auto &lightIndices = workload.drawData.lightIndices;
        auto &lightCounts = workload.drawData.lightCounts;
        auto &lookAtIndices = workload.drawData.lookAtIndices;
        auto &modifyPosUints = workload.drawData.modifyPosUints;
        auto &posScreen = workload.drawData.posScreen;
        uint32_t globalIndex = indices[dstIndex];
        if (used[dstIndex]) {
            auto &posShorts = workload.drawData.posShorts;
            auto &velShorts = workload.drawData.velShorts;
            auto &viewProjIndices = workload.drawData.viewProjIndices;
            auto &worldIndices = workload.drawData.worldIndices;
            auto &posTransformed = workload.drawData.posTransformed;
            const uint32_t newIndex = workload.drawData.vertexCount();
            posShorts.emplace_back(posShorts[globalIndex * 3 + 0]);
            posShorts.emplace_back(posShorts[globalIndex * 3 + 1]);
            posShorts.emplace_back(posShorts[globalIndex * 3 + 2]);
            velShorts.emplace_back(velShorts[globalIndex * 3 + 0]);
            velShorts.emplace_back(velShorts[globalIndex * 3 + 1]);
            velShorts.emplace_back(velShorts[globalIndex * 3 + 2]);
            normColBytes.emplace_back(normColBytes[globalIndex * 4 + 0]);
            normColBytes.emplace_back(normColBytes[globalIndex * 4 + 1]);
            normColBytes.emplace_back(normColBytes[globalIndex * 4 + 2]);
            normColBytes.emplace_back(normColBytes[globalIndex * 4 + 3]);
            tcFloats.emplace_back(tcFloats[globalIndex * 2 + 0]);
            tcFloats.emplace_back(tcFloats[globalIndex * 2 + 1]);
            viewProjIndices.emplace_back(viewProjIndices[globalIndex]);
            worldIndices.emplace_back(worldIndices[globalIndex]);
            fogIndices.emplace_back(fogIndices[globalIndex]);
            lightIndices.emplace_back(lightIndices[globalIndex]);
            lightCounts.emplace_back(lightCounts[globalIndex]);
            lookAtIndices.emplace_back(lookAtIndices[globalIndex]);
            posTransformed.emplace_back(posTransformed[globalIndex]);
            posScreen.emplace_back(posScreen[globalIndex]);
            indices[dstIndex] = newIndex;
            used[dstIndex] = false;
            globalIndex = indices[dstIndex];
        }

        // Modify the attributes.
        switch (dstAttribute) {
        case G_MWO_POINT_RGBA: {
            normColBytes[globalIndex * 4 + 0] = (value >> 24) & 0xFF;
            normColBytes[globalIndex * 4 + 1] = (value >> 16) & 0xFF;
            normColBytes[globalIndex * 4 + 2] = (value >> 8) & 0xFF;
            normColBytes[globalIndex * 4 + 3] = value & 0xFF;
            fogIndices[globalIndex] = 0;
            lightIndices[globalIndex] = 0;
            lightCounts[globalIndex] = 0;
            break;
        }
        case G_MWO_POINT_ST: {
            const float s = int16_t((value >> 16) & 0xFFFF) / 32.0f;
            const float t = int16_t(value & 0xFFFF) / 32.0f;
            tcFloats[globalIndex * 2 + 0] = s;
            tcFloats[globalIndex * 2 + 1] = t;
            lookAtIndices[globalIndex] = 0;
            break;
        }
        case G_MWO_POINT_XYSCREEN: {
            // First bit being 0 indicates it should modify only XY.
            modifyPosUints.emplace_back(globalIndex << 1);
            modifyPosUints.emplace_back(value);

            // We decode on the CPU anyway for draw area tracking and the debugger.
            const uint16_t extX = (value >> 16) & 0xFFFF;
            const uint16_t extY = value & 0xFFFF;
            posScreen[globalIndex][0] = int16_t(extX) / 4.0f;
            posScreen[globalIndex][1] = int16_t(extY) / 4.0f;
            break;
        }
        case G_MWO_POINT_ZSCREEN: {
            // First bit being 1 indicates it should modify only Z.
            modifyPosUints.emplace_back((globalIndex << 1) | 0x1);
            modifyPosUints.emplace_back(value);

            // We decode on the CPU anyway for depth tracking, branchZ and the debugger.
            posScreen[globalIndex][2] = value / 65536.0f;
            break;
        }
        default:
            assert(false && "Unsupported modify vertex");
            break;
        }
    }
    
    void RSP::branchZ(uint32_t branchDl, uint16_t vtxIndex, uint32_t zValue, DisplayList **dl) {
        const bool forceBranch = state->ext.enhancementConfig->f3dex.forceBranch || extended.forceBranch;
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        const Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const uint32_t globalIndex = indices[vtxIndex];
        const float screenZ = workload.drawData.posScreen[globalIndex][2] * DepthRange;
        const float zValueFloat = zValue / 65536.0f;
        if (forceBranch || (screenZ < zValueFloat)) {
            const uint32_t rdramAddress = fromSegmentedMasked(branchDl);
            *dl = reinterpret_cast<DisplayList *>(state->fromRDRAM(rdramAddress)) - 1;
        }
    }

    void RSP::branchW(uint32_t branchDl, uint16_t vtxIndex, uint32_t wValue, DisplayList **dl) {
        const bool forceBranch = state->ext.enhancementConfig->f3dex.forceBranch || extended.forceBranch;
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        const Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        const uint32_t globalIndex = indices[vtxIndex];
        const float posW = workload.drawData.posTransformed[globalIndex][3];
        if (forceBranch || (posW < static_cast<float>(wValue))) {
            const uint32_t rdramAddress = fromSegmentedMasked(branchDl);
            *dl = reinterpret_cast<DisplayList *>(state->fromRDRAM(rdramAddress)) - 1;
        }
    }

    void RSP::setGeometryMode(uint32_t mask) {
        geometryModeStack[geometryModeStackSize - 1] |= mask;
        state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
    }

    void RSP::pushGeometryMode() {
        if (geometryModeStackSize < RSP_EXTENDED_STACK_SIZE) {
            geometryModeStack[geometryModeStackSize] = geometryModeStack[geometryModeStackSize - 1];
            geometryModeStackSize++;
        }
    }

    void RSP::popGeometryMode() {
        if (geometryModeStackSize > 1) {
            geometryModeStackSize--;
            state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
        }
    }

    void RSP::clearGeometryMode(uint32_t mask) {
        geometryModeStack[geometryModeStackSize - 1] &= ~mask;
        state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
    }

    void RSP::modifyGeometryMode(uint32_t offMask, uint32_t onMask) {
        uint32_t &geometryMode = geometryModeStack[geometryModeStackSize - 1];
        geometryMode &= offMask;
        geometryMode |= onMask;
        state->updateDrawStatusAttribute(DrawAttribute::GeometryMode);
    }

    void RSP::setObjRenderMode(uint32_t value) {
        objRenderMode = value;
        state->updateDrawStatusAttribute(DrawAttribute::ObjRenderMode);
    }

    void RSP::setViewport(uint32_t address) {
        setViewport(address, extended.global.viewportOrigin, extended.global.viewportOffsetX, extended.global.viewportOffsetY);
    }
    
    void RSP::setViewport(uint32_t address, uint16_t ori, int16_t offx, int16_t offy) {
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        // Experiment: if GE_LOCK_MATRICES is set, also lock viewport to our injection.
        if (getenv("GE_LOCK_MATRICES") != nullptr &&
            (rdramAddress < 0x007F0000 || rdramAddress > 0x007F01FF)) {
            return;
        }
        const Vp_t *vp = reinterpret_cast<const Vp_t *>(state->fromRDRAM(rdramAddress));
        interop::RSPViewport &viewport = viewportStack[viewportStackSize - 1];
        viewport.scale.x = float(vp->vscale[1]) / 4.0f;
        viewport.scale.y = float(vp->vscale[0]) / 4.0f;
        viewport.scale.z = float(vp->vscale[3]) / DepthRange;
        viewport.translate.x = float(state->rdp->movedFromOrigin(vp->vtrans[1], ori) + offx) / 4.0f;
        viewport.translate.y = float(vp->vtrans[0] + offy) / 4.0f;
        viewport.translate.z = float(vp->vtrans[3]) / DepthRange;
        extended.viewportOrigin = ori;
        viewportChanged = true;
        static int vp_log = 0;
        if (++vp_log <= 10) {
            fprintf(stderr, "[RSP::setViewport #%d] seg=0x%08X phys=0x%08X raw_scale=(%d,%d,%d,%d) raw_trans=(%d,%d,%d,%d) -> scale=(%.2f,%.2f,%.2f) trans=(%.2f,%.2f,%.2f)\n",
                vp_log, address, rdramAddress,
                vp->vscale[0], vp->vscale[1], vp->vscale[2], vp->vscale[3],
                vp->vtrans[0], vp->vtrans[1], vp->vtrans[2], vp->vtrans[3],
                (double)viewport.scale.x, (double)viewport.scale.y, (double)viewport.scale.z,
                (double)viewport.translate.x, (double)viewport.translate.y, (double)viewport.translate.z);
        }
    }

    void RSP::pushViewport() {
        if (viewportStackSize < RSP_EXTENDED_STACK_SIZE) {
            viewportStack[viewportStackSize] = viewportStack[viewportStackSize - 1];
            viewportStackSize++;
        }
    }

    void RSP::popViewport() {
        if (viewportStackSize > 1) {
            viewportStackSize--;
            viewportChanged = true;
        }
    }
    
    void RSP::setLight(uint8_t index, uint32_t address) {
        assert((index >= 0) && (index <= RSP_MAX_LIGHTS));
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const uint8_t *data = reinterpret_cast<const uint8_t *>(state->fromRDRAM(rdramAddress));
        memcpy(&lights[index], data, sizeof(Light));
        lightsChanged = true;
    }

    void RSP::setLightColor(uint8_t index, uint32_t value) {
        assert((index >= 0) && (index <= RSP_MAX_LIGHTS));
        auto &rawLight = lights[index].raw;
        rawLight.words[0] = value;
        rawLight.words[1] = value;
        lightsChanged = true;
    }

    void RSP::setLightCount(uint8_t count) {
        lightCount = count;
        lightsChanged = true;
    }

    void RSP::setClipRatio(uint32_t clipRatio) {
        // TODO
    }

    void RSP::setPerspNorm(uint32_t perspNorm) {
        // TODO
    }

    void RSP::setLookAt(uint8_t index, uint32_t address) {
        assert(index < 2);
        const uint32_t rdramAddress = fromSegmentedMasked(address);
        const DirLight *dirLight = reinterpret_cast<const DirLight *>(state->fromRDRAM(rdramAddress));
        auto &dstLookAt = (index == 1) ? lookAt.y : lookAt.x;
        if ((dirLight->dirx != 0) || (dirLight->diry != 0) || (dirLight->dirz != 0)) {
            dstLookAt = hlslpp::normalize(hlslpp::float3(float(dirLight->dirx), float(dirLight->diry), float(dirLight->dirz)));
        }
        else {
            dstLookAt = { 0.0f, 0.0f, 0.0f };
        }

        lookAtChanged = true;
    }

    void RSP::setLookAtVectors(interop::float3 x, interop::float3 y) {
        lookAt.x = x;
        lookAt.y = y;
        lookAtChanged = true;
    }

    void RSP::setFog(int16_t mul, int16_t offset) {
        fog.mul = mul;
        fog.offset = offset;
        fogChanged = true;
    }
    
    void RSP::setTexture(uint8_t tile, uint8_t level, uint8_t on, uint16_t sc, uint16_t tc) {
        textureState.tile = tile;
        textureState.levels = level + 1;
        textureState.on = on;
        textureState.sc = sc;
        textureState.tc = tc;
    }

    void RSP::setOtherMode(uint32_t high, uint32_t low) {
        interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
        otherMode.H = high;
        otherMode.L = low;
        state->rdp->setOtherMode(otherMode.H, otherMode.L);
    }

    void RSP::pushOtherMode() {
        if (otherModeStackSize < RSP_EXTENDED_STACK_SIZE) {
            otherModeStack[otherModeStackSize] = otherModeStack[otherModeStackSize - 1];
            otherModeStackSize++;
        }
    }

    void RSP::popOtherMode() {
        if (otherModeStackSize > 1) {
            otherModeStackSize--;
            const interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
            state->rdp->setOtherMode(otherMode.H, otherMode.L);
        }
    }

    void RSP::setOtherModeL(uint32_t size, uint32_t off, uint32_t data) {
        interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
        const uint32_t mask = (((uint64_t)(1) << size) - 1) << off;
        otherMode.L = (otherMode.L & (~mask)) | data;
        state->rdp->setOtherMode(otherMode.H, otherMode.L);
    }

    void RSP::setOtherModeH(uint32_t size, uint32_t off, uint32_t data) {
        interop::OtherMode &otherMode = otherModeStack[otherModeStackSize - 1];
        const uint32_t mask = (((uint64_t)(1) << size) - 1) << off;
        otherMode.H = (otherMode.H & (~mask)) | data;
        state->rdp->setOtherMode(otherMode.H, otherMode.L);
    }

    void RSP::setColorImage(uint8_t fmt, uint8_t siz, uint16_t width, uint32_t segAddress) {
        uint32_t phys = fromSegmented(segAddress);
        // Experiment: if GE_FORCE_FB_VI_ORIGIN is set, ALWAYS rewrite CIMG to use the
        // current VI_ORIGIN (the address the VI will display), 320×240 RGBA16. This
        // makes RT64 track framebuffer at the address VI is actually showing, so
        // present can find it.
        // Under GE_DEEP_SHADOW, rewrite every SETCIMG to point at the CURRENT VI_ORIGIN.
        // This makes RT64 render to the same FB the VI will display. Without this, game
        // tracks FBs at garbage SETCIMG addresses (0x00FFFEFF etc.) while VI shows
        // 0x00026100 — no overlap, nothing visible.
        if (getenv("GE_DEEP_SHADOW") != nullptr) {
            uint32_t safe_phys = VI_ORIGIN_REG & 0x00FFFFFF;
            if (safe_phys == 0 || safe_phys >= 0x00800000) safe_phys = 0x00026100;
            static int ci_log2 = 0;
            if (++ci_log2 <= 10) {
                fprintf(stderr, "[setColorImage VI-SYNC #%d] game fmt=%u siz=%u w=%u addr=0x%08X -> fmt=0 siz=2 w=320 addr=0x%08X\n",
                    ci_log2, fmt, siz, width, phys, safe_phys);
            }
            fmt = 0; siz = 2; width = 320; phys = safe_phys;
        }
        else if (getenv("GE_FORCE_FB_VI_ORIGIN") != nullptr) {
            // Observed VI origin addresses: 0x00026100 (primary in-game FB),
            // 0x003DAA80, 0x003B5280, 0x00000900 (early boot). Use 0x00026100 as default.
            uint32_t safe_phys = 0x00026100;
            if (safe_phys == 0 || safe_phys >= 0x00800000) safe_phys = 0x00026100;
            static int ci_log = 0;
            if (++ci_log <= 10) {
                fprintf(stderr, "[RSP::setColorImage FORCE_VI #%d] game fmt=%u siz=%u w=%u addr=0x%08X -> forcing fmt=0 siz=2 w=320 addr=0x%08X\n",
                    ci_log, fmt, siz, width, phys, safe_phys);
            }
            fmt = 0; siz = 2; width = 320; phys = safe_phys;
        }
        else if (getenv("GE_LOCK_MATRICES") != nullptr &&
            (width <= 1 || width > 640 || phys >= 0x00800000)) {
            static int ci_log = 0;
            uint32_t safe_phys = 0x00026100;
            if (++ci_log <= 10) {
                fprintf(stderr, "[RSP::setColorImage OVERRIDE #%d] game_sent fmt=%u siz=%u width=%u addr=0x%08X -> forcing fmt=0 siz=2 width=320 addr=0x%08X\n",
                    ci_log, fmt, siz, width, phys, safe_phys);
            }
            fmt = 0; siz = 2; width = 320; phys = safe_phys;
        }
        state->rdp->setColorImage(fmt, siz, width, phys);
    }

    void RSP::setDepthImage(uint32_t segAddress) {
        state->rdp->setDepthImage(fromSegmented(segAddress));
    }

    void RSP::setTextureImage(uint8_t fmt, uint8_t siz, uint16_t width, uint32_t segAddress) {
        uint32_t phys = fromSegmented(segAddress);
        uint32_t remapped = ge_remap_to_shadow(phys);
        static int ti_log = 0;
        if (++ti_log <= 20) {
            fprintf(stderr, "[setTextureImage #%d] fmt=%u siz=%u w=%u seg=0x%08X phys=0x%08X%s\n",
                ti_log, fmt, siz, width, segAddress, remapped,
                remapped != phys ? " (remapped)" : "");
        }
        state->rdp->setTextureImage(fmt, siz, width, remapped);
    }

    void RSP::drawIndexedTri(uint32_t a, uint32_t b, uint32_t c, bool rawGlobalIndices) {
        static uint32_t tri_count = 0;
        static uint32_t tri_culled = 0;
        tri_count++;

        // Copy mode is not supported when drawing regular tris and crashes the hardware.
        const uint32_t cycleType = state->rdp->otherMode.cycleType();
        assert(cycleType != G_CYC_COPY);

        // Don't draw anything if both tris are being culled.
        const uint32_t &geometryMode = geometryModeStack[geometryModeStackSize - 1];
        if ((geometryMode & cullBothMask) == cullBothMask) {
            tri_culled++;
            return;
        }

        // Log total counts periodically so we can see cull ratio.
        if ((tri_count % 100) == 0) {
            fprintf(stderr, "[drawIndexedTri stats] total=%u culled=%u drawn=%u\n",
                tri_count, tri_culled, tri_count - tri_culled);
        }
        
        state->rdp->checkFramebufferPair();
        
        // We must add the current projection again if we're not in the right state.
        const int workloadCursor = state->ext.workloadQueue->writeCursor;
        Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];
        FramebufferPair &fbPair = workload.fbPairs[workload.currentFramebufferPairIndex()];
        const Projection::Type projType = getCurrentProjectionType();
        if (!fbPair.inProjection(curViewProjIndex, projType)) {
            state->flush();
            addCurrentProjection(projType);
        }

        // Check if the texture needs to be updated.
        auto &drawCall = state->drawCall;
        if ((drawCall.textureOn != textureState.on) || (drawCall.textureTile != textureState.tile) || (drawCall.textureLevels != textureState.levels)) {
            drawCall.textureOn = textureState.on;
            drawCall.textureTile = textureState.tile;
            drawCall.textureLevels = textureState.levels;
            state->updateDrawStatusAttribute(DrawAttribute::Texture);
        }

        if (state->checkDrawState()) {
            state->loadDrawState();
        }

        const bool usesShade = geometryMode & G_SHADE;
        const bool usesLighting = geometryMode & G_LIGHTING;
        const bool usesFog = geometryMode & G_FOG;
        const bool computeSmoothNormals = !usesLighting;

        // Swap the indices around if and only if front face culling is enabled.
        if ((geometryMode & cullBothMask) == cullFrontMask) {
            uint8_t swap = c;
            c = a;
            a = swap;
        }
        
        auto &faceIndices = workload.drawData.faceIndices;
        auto &viewProjIndices = workload.drawData.viewProjIndices;
        auto &worldIndices = workload.drawData.worldIndices;
        auto &posScreen = workload.drawData.posScreen;
        auto &tcFloats = workload.drawData.tcFloats;
        auto &minMatrix = drawCall.minWorldMatrix;
        auto &maxMatrix = drawCall.maxWorldMatrix;
        uint32_t globalIndices[3];
        if (rawGlobalIndices) {
            globalIndices[0] = a;
            globalIndices[1] = b;
            globalIndices[2] = c;
        }
        else {
            globalIndices[0] = indices[a];
            globalIndices[1] = indices[b];
            globalIndices[2] = indices[c];
            used[a] = used[b] = used[c] = true;
        }

        // Indicates the vertex has been used in a tri. Whatever routines modify the vertex afterwards must use a new index instead.

        for (int i = 0; i < 3; i++) {
            // TODO: Figure out how to handle texcoord tracking on TEXGEN cases.
            const uint32_t globalIndex = globalIndices[i];
            state->rdp->updateCallTexcoords(tcFloats[globalIndex * 2 + 0], tcFloats[globalIndex * 2 + 1]);
            faceIndices.push_back(globalIndices[i]);
            minMatrix = std::min(minMatrix, worldIndices[globalIndex]);
            maxMatrix = std::max(maxMatrix, worldIndices[globalIndex]);
        }

        bool visibleTri = true;
        const bool usesCulling = geometryMode & cullBothMask;
        if (usesCulling) {
            const hlslpp::float3 U = posScreen[globalIndices[1]] - posScreen[globalIndices[0]];
            const hlslpp::float3 V = posScreen[globalIndices[2]] - posScreen[globalIndices[0]];
            const hlslpp::float3 N = hlslpp::cross(V, U);
            visibleTri = (N.z >= 0.0f);
        }
        // Debug: dump final screen-space coords of first 20 tris
        static int tri_dump = 0;
        if (++tri_dump <= 20) {
            const hlslpp::float3 &v0 = posScreen[globalIndices[0]];
            const hlslpp::float3 &v1 = posScreen[globalIndices[1]];
            const hlslpp::float3 &v2 = posScreen[globalIndices[2]];
            fprintf(stderr, "[drawTri #%d] visible=%d v0=(%.1f,%.1f,%.3f) v1=(%.1f,%.1f,%.3f) v2=(%.1f,%.1f,%.3f)\n",
                tri_dump, visibleTri,
                (double)v0.x, (double)v0.y, (double)v0.z,
                (double)v1.x, (double)v1.y, (double)v1.z,
                (double)v2.x, (double)v2.y, (double)v2.z);
        }

        const FixedRect &scissorRect = state->rdp->scissorRectStack[state->rdp->scissorStackSize - 1];
        if (visibleTri && !scissorRect.isNull()) {
            fbPair.scissorRect.merge(scissorRect);

            FixedRect drawRect;
            for (int i = 0; i < 3; i++) {
                const hlslpp::float3 &v = posScreen[globalIndices[i]];
                drawRect.ulx = std::min(drawRect.ulx, int32_t(v[0] * 4.0f));
                drawRect.uly = std::min(drawRect.uly, int32_t(v[1] * 4.0f));
                drawRect.lrx = std::max(drawRect.lrx, int32_t(hlslpp::ceil(v.x).x * 4.0f));
                drawRect.lry = std::max(drawRect.lry, int32_t(hlslpp::ceil(v.y).x * 4.0f));
            }

            drawRect = scissorRect.intersection(drawRect);
            if (!drawRect.isNull()) {
                fbPair.drawColorRect.merge(drawRect);
                if (otherModeStack[otherModeStackSize - 1].zUpd()) {
                    fbPair.drawDepthRect.merge(drawRect);
                }
            }
        }

        // 2026-05-04: triangle area histogram. Categorize each tri so we can see
        // how many would actually rasterize visible pixels.
        static thread_local uint32_t tri_zero_area = 0;
        static thread_local uint32_t tri_tiny = 0;        // <16 px²
        static thread_local uint32_t tri_small = 0;       // 16..256 px²
        static thread_local uint32_t tri_medium = 0;      // 256..4096 px²
        static thread_local uint32_t tri_large = 0;       // >4096 px²
        static thread_local uint32_t tri_offscreen = 0;   // bbox outside [0..320]x[0..240]
        static thread_local uint32_t tri_grand_total = 0;
        if (visibleTri) {
            const hlslpp::float3 &p0 = posScreen[globalIndices[0]];
            const hlslpp::float3 &p1 = posScreen[globalIndices[1]];
            const hlslpp::float3 &p2 = posScreen[globalIndices[2]];
            const float ax = p1.x - p0.x, ay = p1.y - p0.y;
            const float bx = p2.x - p0.x, by = p2.y - p0.y;
            const float area = std::abs(ax * by - ay * bx) * 0.5f;
            const float minx = std::min({p0.x, p1.x, p2.x});
            const float maxx = std::max({p0.x, p1.x, p2.x});
            const float miny = std::min({p0.y, p1.y, p2.y});
            const float maxy = std::max({p0.y, p1.y, p2.y});
            const bool onscreen = (maxx >= 0 && minx <= 320 && maxy >= 0 && miny <= 240);
            if (!onscreen) tri_offscreen++;
            else if (area < 1.0f) tri_zero_area++;
            else if (area < 16.0f) tri_tiny++;
            else if (area < 256.0f) tri_small++;
            else if (area < 4096.0f) tri_medium++;
            else tri_large++;
            tri_grand_total++;
            if ((tri_grand_total % 500) == 0) {
                fprintf(stderr, "[tri histogram total=%u] offscreen=%u zero=%u tiny=%u small=%u medium=%u large=%u\n",
                    tri_grand_total, tri_offscreen, tri_zero_area, tri_tiny, tri_small, tri_medium, tri_large);
            }
        }

        drawCall.triangleCount++;
    }

    void RSP::drawIndexedTri(uint32_t a, uint32_t b, uint32_t c) {
        drawIndexedTri(a, b, c, false);
    }

    void RSP::setViewportAlign(uint16_t ori, int16_t offx, int16_t offy) {
        extended.global.viewportOrigin = ori;
        extended.global.viewportOffsetX = offx;
        extended.global.viewportOffsetY = offy;
    }

    void RSP::vertexTestZ(uint8_t vtxIndex) {
        extended.drawExtendedType = DrawExtendedType::VertexTestZ;
        extended.drawExtendedData.vertexTestZ.vertexIndex = indices[vtxIndex];
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
        drawIndexedTri(vtxIndex, vtxIndex, vtxIndex, false);
        extended.drawExtendedType = DrawExtendedType::None;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
    }

    void RSP::endVertexTestZ() {
        uint32_t vtxIndex = extended.drawExtendedData.vertexTestZ.vertexIndex;
        extended.drawExtendedType = DrawExtendedType::EndVertexTestZ;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
        drawIndexedTri(vtxIndex, vtxIndex, vtxIndex, true);
        extended.drawExtendedType = DrawExtendedType::None;
        state->updateDrawStatusAttribute(DrawAttribute::ExtendedType);
    }

    void RSP::matrixId(uint32_t id, bool push, bool proj, bool decompose, uint8_t pos, uint8_t rot, uint8_t scale, uint8_t skew, uint8_t persp, uint8_t vert, uint8_t tile, uint8_t order, uint8_t editable, bool idIsAddress, bool editGroup) {
        assert((idIsAddress == editGroup) && "This case is not supported yet.");

        auto setGroupProperties = [=](TransformGroup* dstGroup, bool newGroup) {
            if (newGroup || (dstGroup->editable == G_EX_EDIT_ALLOW)) {
                dstGroup->decompose = decompose;
                dstGroup->positionInterpolation = pos;
                dstGroup->rotationInterpolation = rot;
                dstGroup->scaleInterpolation = scale;
                dstGroup->skewInterpolation = skew;
                dstGroup->perspectiveInterpolation = persp;
                dstGroup->vertexInterpolation = vert;
                dstGroup->tileInterpolation = tile;
                dstGroup->ordering = order;
                dstGroup->editable = editable;
            }
        };

        if (idIsAddress && editGroup) {
            const uint32_t rdramAddress = fromSegmentedMasked(id);
            const int workloadCursor = state->ext.workloadQueue->writeCursor;
            Workload &workload = state->ext.workloadQueue->workloads[workloadCursor];

            auto range = workload.physicalAddressTransformMap.equal_range(rdramAddress);
            for (auto it = range.first; it != range.second; it++) {
                uint32_t matrix_id = it->second;
                if (proj && (matrix_id < workload.drawData.viewProjTransformGroups.size())) {
                    uint32_t groupIndex = workload.drawData.viewProjTransformGroups[matrix_id];
                    setGroupProperties(&workload.drawData.transformGroups[groupIndex], false);
                }
                else if (matrix_id < workload.drawData.worldTransformGroups.size()) {
                    uint32_t groupIndex = workload.drawData.worldTransformGroups[matrix_id];
                    setGroupProperties(&workload.drawData.transformGroups[groupIndex], false);
                }
            }
        }
        else {
            auto &stack = proj ? extended.viewProjMatrixIdStack : extended.modelMatrixIdStack;
            int &stackSize = proj ? extended.viewProjMatrixIdStackSize : extended.modelMatrixIdStackSize;
            bool &stackChanged = proj ? extended.viewProjMatrixIdStackChanged : extended.modelMatrixIdStackChanged;
            if (push) {
                if (size_t(stackSize) < stack.size()) {
                    stackSize++;
                }
                else {
                    assert(false && "Stack is full.");
                }
            }

            const int stackIndex = stackSize - 1;
            TransformGroup* dstGroup = &stack[stackIndex];
            dstGroup->matrixId = id;
            setGroupProperties(dstGroup, true);
            stackChanged = true;
        }
    }

    void RSP::popMatrixId(uint8_t count, bool proj) {
        int &stackSize = proj ? extended.viewProjMatrixIdStackSize : extended.modelMatrixIdStackSize;
        bool &stackChanged = proj ? extended.viewProjMatrixIdStackChanged : extended.modelMatrixIdStackChanged;
        assert((count <= stackSize) && "Pop has requested a larger amount of matrices than the ones present in the stack.");
        while ((count > 0) && (stackSize > 1)) {
            count--;
            stackSize--;
            stackChanged = true;
        }
    }

    void RSP::forceBranch(bool force) {
        extended.forceBranch = force;
    }

    void RSP::clearExtended() {
        extended.drawExtendedType = DrawExtendedType::None;
        extended.drawExtendedData = {};
        extended.viewportOrigin = G_EX_ORIGIN_NONE;
        extended.global.viewportOrigin = G_EX_ORIGIN_NONE;
        extended.global.viewportOffsetX = 0;
        extended.global.viewportOffsetY = 0;
        extended.modelMatrixIdStack[0] = TransformGroup();
        extended.modelMatrixIdStackSize = 1;
        extended.modelMatrixIdStackChanged = false;
        extended.curModelMatrixIdGroupIndex = 0;
        extended.viewProjMatrixIdStack[0] = TransformGroup();
        extended.viewProjMatrixIdStackSize = 1;
        extended.viewProjMatrixIdStackChanged = false;
        extended.curViewProjMatrixIdGroupIndex = 0;
        extended.forceBranch = false;
    }

    void RSP::setGBI(GBI *gbi) {
        NoN = gbi->flags.NoN;
        cullBothMask = gbi->constants[F3DENUM::G_CULL_BOTH];
        cullFrontMask = gbi->constants[F3DENUM::G_CULL_FRONT];
        projMask = gbi->constants[F3DENUM::G_MTX_PROJECTION];
        loadMask = gbi->constants[F3DENUM::G_MTX_LOAD];
        pushMask = gbi->constants[F3DENUM::G_MTX_PUSH];
        shadingSmoothMask = gbi->constants[F3DENUM::G_SHADING_SMOOTH];
    }
};