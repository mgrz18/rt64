//
// RT64
//

#include "rt64_gbi_f3dgolden.h"

#include "hle/rt64_interpreter.h"

#include "rt64_gbi_f3d.h"
#include "rt64_gbi_f3dex2.h"
#include "rt64_gbi_extended.h"

#include "hle/rt64_rdp.h"
#include "hle/rt64_rsp.h"
#include "hle/rt64_state.h"

namespace RT64 {
    namespace GBI_F3DGOLDEN {
        // 2026-05-04: Per gmain.s disasm, 0xBD is a *bitfield-MOVEWORD* (masked
        // partial-word write), not a plain MOVEWORD. Treating it as plain MOVEWORD
        // overwrites entire DMEM words, corrupting neighboring state (likely segment
        // table → "matrices look like garbage at 0x00264500" symptom). This handler
        // is the experimental baseline: gate via env GE_NO_BD=1 to no-op and see if
        // rendering changes vs the plain-MOVEWORD path.
        void moveWord_BD(State *state, DisplayList **dl) {
            static int log_count = 0;
            if (++log_count <= 20) {
                fprintf(stderr, "[BD #%d] w0=0x%08X w1=0x%08X\n",
                    log_count, (*dl)->w0, (*dl)->w1);
            }
            if (getenv("GE_NO_BD") != nullptr) {
                return;  // no-op gate — drops the write entirely
            }
            GBI_F3D::moveWord(state, dl);
        }

        // GoldenEye's custom triangle opcode at 0xB1 — pack up to 4 indexed triangles
        // into a single command (nibble-packed indices from w0/w1).
        // Verbatim from GLideN64 F3DGOLDEN.cpp:
        //   while (w1 != 0) {
        //       v0 = w1 & 0xF; w1 >>= 4;
        //       v1 = w1 & 0xF; w1 >>= 4;
        //       v2 = w0 & 0xF; w0 >>= 4;
        //       gSPTriangle(v0, v1, v2);
        //   }
        // Indices are raw (NOT divided by 10 like F3D_Tri1). Max 16 vertices (4-bit).
        void triX(State *state, DisplayList **dl) {
            uint32_t w0 = (*dl)->w0;
            uint32_t w1 = (*dl)->w1;
            static int tcx = 0;
            if (++tcx <= 10) {
                fprintf(stderr, "[triX #%d] w0=0x%08X w1=0x%08X\n", tcx, w0, w1);
            }
            while (w1 != 0) {
                uint32_t v0 = w1 & 0xf;
                w1 >>= 4;
                uint32_t v1 = w1 & 0xf;
                w1 >>= 4;
                uint32_t v2 = w0 & 0xf;
                w0 >>= 4;
                state->rsp->drawIndexedTri(v0, v1, v2);
            }
        }

        void setup(GBI *gbi) {
            // **2026-04-21**: Per GLideN64 source (gonetz/GLideN64 F3DGOLDEN.cpp),
            // GoldenEye uses **standard Fast3D (gspFast3DNoN) opcodes** with only TWO
            // deviations:
            //   - 0xB1 = G_TRIX (GE custom packed triangle) — was G_QUAD in F3D
            //   - 0xBD = G_MOVEWORD (remapped from POPMTX — GE never pops matrix stack)
            //
            // All other opcodes are standard F3D:
            //   0x00 G_SPNOOP    0x01 G_MTX       0x03 G_MOVEMEM   0x04 G_VTX
            //   0x06 G_DL        0x09 G_SPRITE2D  0xB4 G_RDPHALF_1 0xB5 G_QUAD
            //   0xB6 G_CLEARGEOMETRYMODE  0xB7 G_SETGEOMETRYMODE   0xB8 G_ENDDL
            //   0xB9 G_SETOTHERMODE_L     0xBA G_SETOTHERMODE_H    0xBB G_TEXTURE
            //   0xBC G_MOVEWORD  0xBE G_CULLDL   0xBF G_TRI1
            //   0xE0-0xFF: standard RDP commands (SETCIMG, FILLRECT, etc.)
            //
            // Config flags for GE (per GLideN64): NoN=true (skip near-plane clip),
            // negativeY=true, legacyVertexPipeline=true.
            //
            // Critical correction from earlier session: we were wrongly using F3DEX2
            // mappings (0x01 G_VTX, 0x05 G_TRI1, etc.) — those are for F3DEX2 ucode,
            // NOT GoldenEye's F3DGOLDEN. Base Fast3D has 0x01=G_MTX, 0x04=G_VTX,
            // 0xBF=G_TRI1, which is what GoldenEye actually uses.
            GBI_F3D::setup(gbi);

            // 2026-05-04: REVERTED experimental 0x01→G_DL remap. The decomp
            // Makefile does NOT define F3DEX_GBI, so gbi.h uses stock F3D defines
            // (G_MTX=0x01, G_DL=0x06). The asm finding that "0x01 acts like
            // a DL pointer update" was either a sub-opcode dispatch artifact or
            // the agent misidentified the handler. Leaving the standard F3D
            // mappings until we can pin down the exact dispatch table.
            gbi->map[F3DGOLDEN_G_TRIX]     = &triX;               // 0xB1 (was QUAD in F3D)
            gbi->map[F3DGOLDEN_G_MOVEWORD] = &moveWord_BD;        // 0xBD (was POPMTX)

            // GE emits SOME F3DEX2-style RDP commands (observed 0xE3=179/DL). Agent's
            // report suggested everything is standard F3D but empirical histogram shows
            // 0xE2/0xE3 are used. Install F3DEX2-style handlers for these RDP-range ops
            // that the game actually emits; they're harmless for std F3D opcode range.
            gbi->map[F3DEX2_G_SETOTHERMODE_H] = &GBI_F3DEX2::setOtherModeH;  // 0xE3
            gbi->map[F3DEX2_G_SETOTHERMODE_L] = &GBI_F3DEX2::setOtherModeL;  // 0xE2
        }
    }
};
