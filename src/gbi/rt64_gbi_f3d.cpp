//
// RT64
//

#include "rt64_gbi_f3d.h"

#include <cassert>
#include <unordered_map>
#include <cstring>

#include "../include/rt64_extended_gbi.h"

#include "rt64_f3d.h"
#include "rt64_gbi_extended.h"
#include "rt64_gbi_rdp.h"

namespace RT64 {
    namespace GBI_F3D {
        void matrix(State *state, DisplayList **dl) {
            static int mlog = 0;
            if (++mlog <= 10) {
                fprintf(stderr, "[F3D::matrix cmd #%d] w0=0x%08X w1=0x%08X (p@16-23=0x%02X p@0-7=0x%02X ofs8-15=0x%02X)\n",
                    mlog, (*dl)->w0, (*dl)->w1,
                    (*dl)->p0(16, 8), (*dl)->p0(0, 8), (*dl)->p0(8, 8));
            }
            state->rsp->matrix((*dl)->w1, (*dl)->p0(16, 8));
        }

        void popMatrix(State *state, DisplayList **dl) {
            if ((*dl)->w1 == 0) {
                state->rsp->popMatrix(1);
            }
        }
        
        void moveMem(State *state, DisplayList **dl) {
            switch ((*dl)->p0(16, 8)) {
            case F3D_G_MV_VIEWPORT:
                state->rsp->setViewport((*dl)->w1);
                break;
            case F3D_G_MV_MATRIX_1:
                state->rsp->forceMatrix((*dl)->w1);
                *dl = *dl + 3;
                break;
            case F3D_G_MV_L0:
                state->rsp->setLight(0, (*dl)->w1);
                break;
            case F3D_G_MV_L1:
                state->rsp->setLight(1, (*dl)->w1);
                break;
            case F3D_G_MV_L2:
                state->rsp->setLight(2, (*dl)->w1);
                break;
            case F3D_G_MV_L3:
                state->rsp->setLight(3, (*dl)->w1);
                break;
            case F3D_G_MV_L4:
                state->rsp->setLight(4, (*dl)->w1);
                break;
            case F3D_G_MV_L5:
                state->rsp->setLight(5, (*dl)->w1);
                break;
            case F3D_G_MV_L6:
                state->rsp->setLight(6, (*dl)->w1);
                break;
            case F3D_G_MV_L7:
                state->rsp->setLight(7, (*dl)->w1);
                break;
            case F3D_G_MV_LOOKATX:
                state->rsp->setLookAt(0, (*dl)->w1);
                break;
            case F3D_G_MV_LOOKATY:
                state->rsp->setLookAt(1, (*dl)->w1);
                break;
            default:
                assert(false && "Unimplemented move mem.");
                break;
            }
        }
        
        void vertex(State *state, DisplayList **dl) {
            state->rsp->setVertex((*dl)->w1, (*dl)->p0(20, 4) + 1, (*dl)->p0(16, 4));
        }

        void runDl(State *state, DisplayList **dl) {
            static int rundl_log = 0;
            if (++rundl_log <= 15) {
                fprintf(stderr, "[runDl #%d] w0=0x%08X w1=0x%08X\n", rundl_log, (*dl)->w0, (*dl)->w1);
            }

            uint32_t rdramAddress = state->rsp->fromSegmentedMasked((*dl)->w1);
            if (rundl_log <= 15) {
                fprintf(stderr, "  resolved phys=0x%08X\n", rdramAddress);
            }

            // Guard: if target is in OS/boot area (< 0x1000) or otherwise suspicious, skip
            // the G_DL instead of walking into uninitialized memory. Prevents SAFETY_ABORT
            // when game emits G_DL to sentinel addresses like 0x80000000 (seg 8 = 0 base).
            if (rdramAddress < 0x1000 || rdramAddress >= 0x00800000) {
                static int skip_log = 0;
                if (++skip_log <= 10) {
                    fprintf(stderr, "[runDl SKIP] bad target w1=0x%08X phys=0x%08X — skipping G_DL\n", (*dl)->w1, rdramAddress);
                }
                return;  // Don't advance dl; outer loop will dl++ to next cmd.
            }

            if ((*dl)->p0(16, 1) == 0) {
                state->pushReturnAddress(*dl);
            }

            // GE_LAZY_SHADOW: when G_DL jumps to a game heap address that may have been
            // overwritten, copy a chunk to a lazy shadow area and redirect. Heuristic:
            // addresses in the game's typical DL heap range (0x00200000..0x00400000 phys
            // = 0x80200000..0x80400000 virt). Lazy shadow sits at 0x00700000, 1MB worth
            // of 16KB chunks (64 chunks). Deduplicate via static map.
            if (getenv("GE_LAZY_SHADOW") != nullptr) {
                constexpr uint32_t HEAP_LO = 0x00200000;
                constexpr uint32_t HEAP_HI = 0x00400000;
                constexpr uint32_t LAZY_BASE = 0x00700000;
                constexpr uint32_t LAZY_SIZE = 0x00100000;  // 1MB
                constexpr uint32_t CHUNK_SIZE = 0x10000;    // 64KB chunks
                if (rdramAddress >= HEAP_LO && rdramAddress < HEAP_HI) {
                    static std::unordered_map<uint32_t, uint32_t> shadow_map;
                    static uint32_t next_slot = 0;
                    // Align target to chunk to find the chunk containing it.
                    uint32_t chunk_base = rdramAddress & ~(CHUNK_SIZE - 1);
                    auto it = shadow_map.find(chunk_base);
                    uint32_t shadow_chunk;
                    if (it == shadow_map.end()) {
                        // Pre-fill entire lazy shadow area with G_ENDDL sentinels on first
                        // allocation. Walker reading ANYTHING unallocated will hit G_ENDDL
                        // and pop return stack cleanly instead of walking through zeros.
                        static bool lazy_initialized = false;
                        if (!lazy_initialized) {
                            for (uint32_t off = 0; off < LAZY_SIZE; off += 8) {
                                *(uint32_t*)(state->RDRAM + LAZY_BASE + off + 0) = 0xB8000000;
                                *(uint32_t*)(state->RDRAM + LAZY_BASE + off + 4) = 0x00000000;
                            }
                            lazy_initialized = true;
                        }
                        shadow_chunk = LAZY_BASE + (next_slot * CHUNK_SIZE);
                        next_slot = (next_slot + 1) % (LAZY_SIZE / CHUNK_SIZE);
                        memcpy(state->RDRAM + shadow_chunk, state->RDRAM + chunk_base, CHUNK_SIZE);
                        // Last 256 bytes of chunk become sentinels too (overwrite data, but
                        // ensures walker stops if it walks past chunk boundary zero regions
                        // that might coincide with SPNOOP 0x00).
                        for (uint32_t sentinel_off = CHUNK_SIZE - 256; sentinel_off < CHUNK_SIZE; sentinel_off += 8) {
                            *(uint32_t*)(state->RDRAM + shadow_chunk + sentinel_off + 0) = 0xB8000000;
                            *(uint32_t*)(state->RDRAM + shadow_chunk + sentinel_off + 4) = 0x00000000;
                        }
                        shadow_map[chunk_base] = shadow_chunk;
                        static int ls_log = 0;
                        if (++ls_log <= 10) {
                            fprintf(stderr, "[lazy_shadow #%d] chunk 0x%08X -> 0x%08X (target was 0x%08X)\n",
                                ls_log, chunk_base, shadow_chunk, rdramAddress);
                        }
                    } else {
                        shadow_chunk = it->second;
                    }
                    // Redirect to shadow equivalent
                    uint32_t offset_in_chunk = rdramAddress - chunk_base;
                    rdramAddress = shadow_chunk + offset_in_chunk;
                }
            }

            *dl = reinterpret_cast<DisplayList *>(state->fromRDRAM(rdramAddress)) - 1;
        }

        void endDl(State *state, DisplayList **dl) {
            *dl = state->popReturnAddress();
        }

        void sprite2DBase(State *state, DisplayList **dl) {
            // TODO
        }

        void tri1(State *state, DisplayList **dl) {
            state->rsp->drawIndexedTri((*dl)->p1(16, 8) / 10, (*dl)->p1(8, 8) / 10, (*dl)->p1(0, 8) / 10);
        }
        
        void quad(State *state, DisplayList **dl) {
            const uint8_t v0 = (*dl)->p1(24, 8) / 10;
            const uint8_t v1 = (*dl)->p1(16, 8) / 10;
            const uint8_t v2 = (*dl)->p1(8, 8) / 10;
            const uint8_t v3 = (*dl)->p1(0, 8) / 10;
            state->rsp->drawIndexedTri(v0, v1, v2);
            state->rsp->drawIndexedTri(v0, v2, v3);
        }

        void cullDl(State *state, DisplayList **dl) {
            // TODO
        }

        void moveWord(State *state, DisplayList **dl) {
            uint8_t type = (*dl)->p0(0, 8);
            switch (type) {
            case G_MW_MATRIX:
                assert(false);
                // TODO
                break;
            case G_MW_NUMLIGHT:
                state->rsp->setLightCount((((*dl)->w1 - 0x80000000) >> 5) - 1);
                break;
            case G_MW_CLIP:
                // TODO
                break;
            case G_MW_SEGMENT:
                state->rsp->setSegment((*dl)->p0(10, 4), (*dl)->w1);
                break;
            case G_MW_FOG:
                state->rsp->setFog((int16_t)((*dl)->p1(16, 16)), (int16_t)((*dl)->p1(0, 16)));
                break;
            case G_MW_LIGHTCOL:
                switch ((*dl)->p0(8, 16)) {
                case G_MWO_aLIGHT_1:
                    state->rsp->setLightColor(0, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_2:
                    state->rsp->setLightColor(1, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_3:
                    state->rsp->setLightColor(2, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_4:
                    state->rsp->setLightColor(3, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_5:
                    state->rsp->setLightColor(4, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_6:
                    state->rsp->setLightColor(5, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_7:
                    state->rsp->setLightColor(6, (*dl)->w1);
                    break;
                case F3D_G_MWO_aLIGHT_8:
                    state->rsp->setLightColor(7, (*dl)->w1);
                    break;
                }

                break;
            case F3D_G_MW_POINTS: 
                {
                    const uint32_t value = (*dl)->p0(8, 16);
                    state->rsp->modifyVertex(value / 40, value % 40, (*dl)->w1);
                }
                break;
            case G_MW_PERSPNORM:
                // TODO
                break;
            default:
                break;
            }
        }

        void texture(State *state, DisplayList **dl) {
            uint8_t tile = (*dl)->p0(8, 3);
            uint8_t level = (*dl)->p0(11, 3);
            uint8_t on = (*dl)->p0(0, 8);
            uint16_t sc = (*dl)->p1(16, 16);
            uint16_t tc = (*dl)->p1(0, 16);
            state->rsp->setTexture(tile, level, on, sc, tc);
        }

        void setOtherModeH(State *state, DisplayList **dl) {
            state->rsp->setOtherModeH((*dl)->p0(0, 8), (*dl)->p0(8, 8), (*dl)->w1);
        }

        void setOtherModeL(State *state, DisplayList **dl) {
            state->rsp->setOtherModeL((*dl)->p0(0, 8), (*dl)->p0(8, 8), (*dl)->w1);
        }

        void setGeometryMode(State *state, DisplayList **dl) {
            state->rsp->setGeometryMode((*dl)->w1);
        }

        void clearGeometryMode(State *state, DisplayList **dl) {
            state->rsp->clearGeometryMode((*dl)->w1);
        }

        void rdpHalf1(State *state, DisplayList **dl) {
            state->microcode.half1 = (*dl)->w1;
        }

        void rdpHalf2(State *state, DisplayList **dl) {
            state->microcode.half2 = (*dl)->w1;
        }

        void setColorImage(State *state, DisplayList **dl) {
            const uint8_t fmt = (*dl)->p0(21, 3);
            const uint8_t siz = (*dl)->p0(19, 2);
            const uint16_t width = (*dl)->p0(0, 12) + 1;
            const uint32_t address = (*dl)->w1;
            static int log_ctr = 0;
            if (++log_ctr <= 10) {
                fprintf(stderr, "[F3D setColorImage #%d] fmt=%u siz=%u width=%u addr=0x%08X\n",
                    log_ctr, fmt, siz, width, address);
            }
            state->rsp->setColorImage(fmt, siz, width, address);
        }

        void setDepthImage(State *state, DisplayList **dl) {
            const uint32_t address = (*dl)->w1;
            state->rsp->setDepthImage(address);
        }

        void setTextureImage(State *state, DisplayList **dl) {
            const uint8_t fmt = (*dl)->p0(21, 3);
            const uint8_t siz = (*dl)->p0(19, 2);
            const uint16_t width = (*dl)->p0(0, 12) + 1;
            const uint32_t address = (*dl)->w1;
            state->rsp->setTextureImage(fmt, siz, width, address);
        }

        void reset(State *state) {
            state->rsp->setLookAtVectors(hlslpp::float3(0.0f, 1.0f, 0.0f), hlslpp::float3(1.0f, 0.0f, 0.0f));
            state->rsp->setFog(0x0100, 0x0000);
        }

        void setup(GBI *gbi) {
            gbi->constants = {
                { F3DENUM::G_MTX_MODELVIEW, 0x00 },
                { F3DENUM::G_MTX_PROJECTION, 0x01 },
                { F3DENUM::G_MTX_MUL, 0x00 },
                { F3DENUM::G_MTX_LOAD, 0x02 },
                { F3DENUM::G_MTX_NOPUSH, 0x00 },
                { F3DENUM::G_MTX_PUSH, 0x04 },
                { F3DENUM::G_TEXTURE_ENABLE, 0x00000002 },
                { F3DENUM::G_SHADING_SMOOTH, 0x00000200 },
                { F3DENUM::G_CULL_FRONT, 0x00001000 },
                { F3DENUM::G_CULL_BACK, 0x00002000 },
                { F3DENUM::G_CULL_BOTH, 0x00003000 }
            };
            
            gbi->map[F3D_G_SPNOOP] = &GBI_EXTENDED::noOpHook;
            gbi->map[F3D_G_MTX] = &matrix;
            gbi->map[F3D_G_MOVEMEM] = &moveMem;
            gbi->map[F3D_G_VTX] = &vertex;
            gbi->map[F3D_G_DL] = &runDl;
            gbi->map[F3D_G_ENDDL] = &endDl;
            gbi->map[F3D_G_SPRITE2D_BASE] = &sprite2DBase;
            gbi->map[F3D_G_TRI1] = &tri1;
            gbi->map[F3D_G_QUAD] = &quad;
            gbi->map[F3D_G_CULLDL] = &cullDl;
            gbi->map[F3D_G_POPMTX] = &popMatrix;
            gbi->map[F3D_G_MOVEWORD] = &moveWord;
            gbi->map[F3D_G_TEXTURE] = &texture;
            gbi->map[F3D_G_SETOTHERMODE_H] = &setOtherModeH;
            gbi->map[F3D_G_SETOTHERMODE_L] = &setOtherModeL;
            gbi->map[F3D_G_SETGEOMETRYMODE] = &setGeometryMode;
            gbi->map[F3D_G_CLEARGEOMETRYMODE] = &clearGeometryMode;
            gbi->map[F3D_G_RDPHALF_1] = &rdpHalf1;
            gbi->map[F3D_G_RDPHALF_2] = &rdpHalf2;
            gbi->map[G_SETCIMG] = &setColorImage;
            gbi->map[G_SETZIMG] = &setDepthImage;
            gbi->map[G_SETTIMG] = &setTextureImage;
            gbi->map[G_RDPNOOP] = &GBI_RDP::noOp;

            gbi->resetFromTask = &reset;
        }
    }
};