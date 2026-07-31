/*
 * Copyright (C) 2026 KytyPS5 Emulator Project
 * SPDX-License-Identifier: GPL-2.0-or-later
 *
 * Regression coverage for two shader program-termination fixes:
 *
 *  1. S_CODE_END (SOPP opcode 0x1f) ends the shader and pads the last instruction-cache
 *     line. It used to be absent from the SOPP opcode table, so the decoder treated it as
 *     a generic no-op and kept walking through the padding into the trailing data,
 *     misdecoding it as instructions (which produced bogus "unsupported operand" and
 *     "unknown encoding" failures far past the real end of the program).
 *
 *  2. A tail S_SETPC_B64 whose target cannot be resolved to an in-shader address is a
 *     return to a caller-supplied address (the PC comes from a live-in SGPR pair and
 *     nothing executable follows). It must be treated as program end. A NON-tail
 *     unresolvable S_SETPC_B64 must still be rejected, because that may be genuine
 *     computed control flow we cannot model.
 */

#include "graphics/shader/recompiler/ShaderCFG.h"
#include "graphics/shader/recompiler/ShaderDecoder.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {
namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ShaderDecoderTerminationTests: failed: %s\n", text);
		std::abort();
	}
}

// SOPP: ENCODING[31:23] = 10_1111111, OP[22:16], SIMM16[15:0].
constexpr uint32_t Sopp(uint32_t op, uint32_t simm16 = 0) {
	return 0xbf800000u | (op << 16u) | simm16;
}

// SOP1: ENCODING[31:23] = 10_1111101, SDST[22:16], OP[15:8], SSRC0[7:0].
constexpr uint32_t Sop1(uint32_t op, uint32_t ssrc0, uint32_t sdst = 0) {
	return 0xbe800000u | (sdst << 16u) | (op << 8u) | ssrc0;
}

constexpr uint32_t kSNop     = Sopp(0x00u);
constexpr uint32_t kSEndpgm  = Sopp(0x01u);
constexpr uint32_t kSCodeEnd = Sopp(0x1fu);
// s_setpc_b64 s[6:7] - SOP1 opcode 0x20, scalar source 6.
constexpr uint32_t kSSetpcS6 = Sop1(0x20u, 6u);

ShaderRecompiler::Decoder::Program Decode(const std::vector<uint32_t>& code, bool expect_ok) {
	ShaderRecompiler::Decoder::Program program;
	std::string                        error;
	const bool ok = ShaderRecompiler::Decoder::DecodeProgram(std::span {code}, program, &error);
	if (ok != expect_ok) {
		std::fprintf(stderr, "decode ok=%d expected=%d error=%s\n", ok ? 1 : 0, expect_ok ? 1 : 0,
		             error.c_str());
		for (const auto& inst: program.instructions) {
			std::fprintf(stderr, "  pc=0x%x word=0x%08x family=%u opcode_id=0x%x opcode=%u words=%u\n",
			             inst.pc, inst.word, static_cast<uint32_t>(inst.family), inst.opcode_id,
			             static_cast<uint32_t>(inst.opcode), inst.word_count);
		}
		std::abort();
	}
	return program;
}

// The observed real-world failure: after S_CODE_END the guest memory holds non-instruction
// data. Decoding must stop at S_CODE_END and never interpret those words.
void TestCodeEndStopsDecode() {
	const std::vector<uint32_t> code {
	    kSNop, kSCodeEnd,
	    // Trailing padding/data taken from a real shader tail. Previously this was decoded
	    // as instructions and reported bogus failures.
	    0x00000000u, 0x30306c73u, 0x000000c3u, 0xd2f2c71du, 0xc6dce377u,
	};

	const auto program = Decode(code, /*expect_ok=*/true);
	Check(program.instructions.size() == 2, "decode stops at S_CODE_END");
	Check(program.instructions.back().opcode == ShaderRecompiler::Decoder::Opcode::SCodeEnd,
	      "last decoded instruction is S_CODE_END");
}

// S_CODE_END is also a valid program end for the CFG, exactly like S_ENDPGM.
void TestCodeEndBuildsGraph() {
	const std::vector<uint32_t> code {kSNop, kSCodeEnd};
	const auto                  program = Decode(code, /*expect_ok=*/true);

	ShaderRecompiler::CFG::Graph graph;
	std::string                  error;
	const bool ok = ShaderRecompiler::CFG::BuildGraph(program, graph, &error);
	Check(ok && !graph.unsupported, "S_CODE_END terminated program builds a CFG");
}

// A shader ending in s_setpc_b64 from a live-in SGPR is a return; it must not be rejected.
// Note the program still needs a terminator the DECODER recognises (S_CODE_END or S_ENDPGM)
// - the tail-return handling is a CFG concern, not a decode concern - and real shaders always
// have S_CODE_END padding after the return.
void TestTailSetpcIsReturn() {
	for (const uint32_t terminator : {kSCodeEnd, kSEndpgm}) {
		const std::vector<uint32_t> code {kSNop, kSSetpcS6, terminator};
		const auto                  program = Decode(code, /*expect_ok=*/true);

		ShaderRecompiler::CFG::Graph graph;
		std::string                  error;
		const bool ok = ShaderRecompiler::CFG::BuildGraph(program, graph, &error);
		if (!ok || graph.unsupported) {
			std::fprintf(stderr, "tail s_setpc_b64 rejected (terminator=0x%08x): %s\n", terminator,
			             error.c_str());
			std::abort();
		}
	}
}

// ...but an unresolvable s_setpc_b64 in the middle of a program is still unsupported: it
// may be real computed control flow, and silently treating it as a return would drop code.
void TestMidShaderSetpcStillFails() {
	const std::vector<uint32_t> code {kSSetpcS6, kSNop, kSEndpgm};
	const auto                  program = Decode(code, /*expect_ok=*/true);

	ShaderRecompiler::CFG::Graph graph;
	std::string                  error;
	const bool ok = ShaderRecompiler::CFG::BuildGraph(program, graph, &error);
	Check(!ok || graph.unsupported, "non-tail dynamic s_setpc_b64 is still rejected");
}

} // namespace
} // namespace Libs::Graphics

int main() {
	Libs::Graphics::TestCodeEndStopsDecode();
	Libs::Graphics::TestCodeEndBuildsGraph();
	Libs::Graphics::TestTailSetpcIsReturn();
	Libs::Graphics::TestMidShaderSetpcStillFails();
	std::printf("ShaderDecoderTerminationTests: all tests passed\n");
	return 0;
}
