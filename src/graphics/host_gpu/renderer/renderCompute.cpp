#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/descriptorCache.h"
#include "graphics/host_gpu/renderer/descriptors.h"
#include "graphics/host_gpu/renderer/imageInfo.h"
#include "graphics/host_gpu/renderer/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/shaderSubgroup.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ShaderIR.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static uint64_t BufferDescriptorSize(const ShaderBufferResource& descriptor) {
	const uint64_t records = descriptor.NumRecords();
	const uint64_t stride  = descriptor.Stride();
	if (stride != 0 && records > UINT64_MAX / stride) {
		EXIT("compute buffer descriptor footprint overflow\n");
	}
	return stride == 0 ? records : records * stride;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const RenderCommandBuffer&    buffer) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		if (!resource.written && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeImageClear(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& program   = *input.stage.program;
	const auto& resources = *input.stage.resources;
	if (program.info.buffers.size() != 1 || resources.buffers.size() != 1 ||
	    !program.info.images.empty() || !program.info.samplers.empty() ||
	    !program.info.addresses.empty() || !resources.images.empty() ||
	    !resources.samplers.empty() || !resources.addresses.empty()) {
		return false;
	}
	const auto& resource   = program.info.buffers.front();
	const auto& raw        = resources.buffers.front();
	const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(raw);
	if (!resource.formatted || !resource.written || resource.read || resource.atomic ||
	    resource.scalar || resource.max_byte_extent != 16 || descriptor.Stride() != 16 ||
	    descriptor.Format() != Prospero::GpuEnumValue(Prospero::BufferFormat::k32_32_32_32UInt) ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    resource.packed_stride != descriptor.PackedStride() || raw.dword_count != 4 ||
	    program.user_data_base != 0 || resources.user_data.size() != 8) {
		return false;
	}
	for (uint32_t i = 0; i < raw.dword_count; i++) {
		if (raw.dwords[i] != resources.user_data[i]) {
			return false;
		}
	}
	const uint32_t clear = resources.user_data[4];
	if (resources.user_data[5] != clear || resources.user_data[6] != clear ||
	    resources.user_data[7] != clear) {
		return false;
	}
	const bool full_dispatch =
	    input.dispatch_thread_dimensions && input.threads_num[0] == 64 &&
	    input.threads_num[1] == 1 && input.threads_num[2] == 1 && group_x != 0 && group_y == 1 &&
	    group_z == 1 && input.dispatch_threads_num[0] == group_x &&
	    input.dispatch_threads_num[1] == 1 && input.dispatch_threads_num[2] == 1 &&
	    input.group_id[0] && !input.group_id[1] && !input.group_id[2] &&
	    input.thread_ids_num == 1 && input.wave_size == 32 && !input.tg_size_en && mode == 0x61u &&
	    group_x % input.threads_num[0] == 0 && descriptor.NumRecords() == group_x;
	const auto size = BufferDescriptorSize(descriptor);
	if (!full_dispatch || size == 0) {
		return false;
	}
	resolved_descriptor = descriptor;
	resolved_clear      = clear;
	resolved_size       = size;
	return true;
}

static bool TryConsumeComputeImageClear(const ShaderComputeInputInfo& input, CommandBuffer& command,
                                        uint32_t group_x, uint32_t group_y, uint32_t group_z,
                                        uint32_t mode) {
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeImageClear(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	auto& cache = command.GetContext().GetTextureCache();
	if (!cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear)) {
		return false;
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute image clear shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     input.stage.program->shader_hash, descriptor.Base48(), size, packed_clear);
	}
	return true;
}

void RenderExecutor::DispatchDirect(uint64_t submit_id, RenderCommandBuffer& buffer,
                                    uint32_t thread_group_x, uint32_t thread_group_y,
                                    uint32_t thread_group_z, uint32_t mode) {
	EXIT_IF(buffer.IsInvalid());
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DispatchDirect), submit_id,
	                    thread_group_x, thread_group_y, thread_group_z, mode,
	                    sh_ctx.GetCs().cs_regs.data_addr);

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	if (!ShaderAddressValid(sh_ctx.GetCs().cs_regs.data_addr)) {
		// A silently-dropped dispatch leaves no storage-bind trace. If the color-grading LUT
		// generation compute is dropped here, that alone explains the never-written LUT.
		LOGF_BOUNDED(128,
		             "DispatchDirect: SKIPPED dispatch, invalid CS shader addr=0x%016" PRIx64
		             " groups=%ux%ux%u mode=0x%08" PRIx32 "\n",
		             sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		             thread_group_z, mode);
		return;
	}

	// Census: record every dispatch that reaches execution (bounded), so the total dispatch
	// count and shader addresses can be correlated against the missing LUT producer.
	LOGF_BOUNDED(256, "DispatchDirect: census shader=0x%016" PRIx64 " groups=%ux%ux%u mode=0x%08" PRIx32
	             "\n",
	             sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y, thread_group_z,
	             mode);

	constexpr uint32_t DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS = 1u << 5u;
	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS             = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS         = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();
	const auto& sh_regs = ctx.GetShaderRegisters();

	ShaderComputeInputInfo    input_info {};
	std::span<const uint32_t> cs_shader;
	if (!ShaderCompileInfoCS(cs_regs, sh_regs, input_info, cs_shader)) {
		EXIT("ShaderCompileInfoCS failed for dispatch with CS shader 0x%016" PRIx64 "\n",
		     cs_regs.cs_regs.data_addr);
	}

	const bool use_thread_dimensions = (mode & DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	if (use_thread_dimensions) {
		input_info.dispatch_thread_dimensions = true;
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
	const bool     large_workgroup =
	    (input_info.threads_num[0] * input_info.threads_num[1] * input_info.threads_num[2] >= 512);
	const auto& program   = *input_info.stage.program;
	const auto& resources = *input_info.stage.resources;
	if (TryConsumeComputeMetaClear(input_info, buffer)) {
		LOGF_BOUNDED(64, "DispatchDirect: CONSUMED as meta-clear shader=0x%016" PRIx64 "\n",
		             sh_ctx.GetCs().cs_regs.data_addr);
		ResetBindings();
		return;
	}
	if (TryConsumeComputeImageClear(input_info, buffer, thread_group_x, thread_group_y,
	                                thread_group_z, mode)) {
		LOGF_BOUNDED(64, "DispatchDirect: CONSUMED as image-clear shader=0x%016" PRIx64 "\n",
		             sh_ctx.GetCs().cs_regs.data_addr);
		ResetBindings();
		return;
	}
	const auto sampled_images = std::count_if(
	    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
		    return image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
		           image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint;
	    });
	const bool                   has_sampler = !program.info.samplers.empty();
	// Targeted, uncapped detector for the color-grading LUT producer. Any WRITTEN storage image
	// whose extent is 32x32x32 (3D volume or 32-layer array) is the CombineLUTs output we have
	// been unable to locate; log it regardless of dispatch count so a late menu-time generation
	// is not lost to the bounded census above. Shape-based (not format-based) to stay robust.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto& image   = program.info.images[i];
		const bool  storage = image.kind == ShaderRecompiler::IR::ResourceKind::StorageImage ||
		                      image.kind == ShaderRecompiler::IR::ResourceKind::StorageImageUint;
		if (!image.written && !storage) {
			continue;
		}
		const auto     r = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
		const uint32_t w = static_cast<uint32_t>(r.Width5()) + 1u;
		const uint32_t h = static_cast<uint32_t>(r.Height5()) + 1u;
		const uint32_t d = static_cast<uint32_t>(r.Depth()) + 1u;
		if (w == 32u && h == 32u && d == 32u) {
			LOGF("DispatchDirect: LUT-SHAPED STORAGE WRITE shader=0x%016" PRIx64
			     " target=0x%010" PRIx64 " type=%u fmt=%u %ux%ux%u tile=%u written=%d storage=%d\n",
			     sh_ctx.GetCs().cs_regs.data_addr, r.Base40(), static_cast<uint32_t>(r.Type()),
			     r.Format(), w, h, d, r.TileMode(), image.written ? 1 : 0, storage ? 1 : 0);
		}
	}
	static std::atomic<uint32_t> dispatch_log_count {0};
	if ((large_workgroup || has_sampler) &&
	    dispatch_log_count.fetch_add(1, std::memory_order_relaxed) < 512) {
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x, thread_group_y,
		     thread_group_z, mode, input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(), program.bindings.push_constant_size);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.Format());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     (image.kind == ShaderRecompiler::IR::ResourceKind::Image ||
			      image.kind == ShaderRecompiler::IR::ResourceKind::ImageUint)
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), r.Format(),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     std::max<uint32_t>(static_cast<uint32_t>(r.LastLevel()),
			                        static_cast<uint32_t>(r.MaxMip())) +
			         1u,
			     r.TileMode());
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(resources.samplers[i]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	if (use_thread_dimensions) {
		auto groups_from_threads = [](uint32_t threads, uint32_t group_size) {
			return (threads == 0
			            ? 0u
			            : (threads + std::max(group_size, 1u) - 1u) / std::max(group_size, 1u));
		};

		const uint32_t old_x = thread_group_x;
		const uint32_t old_y = thread_group_y;
		const uint32_t old_z = thread_group_z;
		thread_group_x       = groups_from_threads(thread_group_x, cs_regs.cs_regs.num_thread_x);
		thread_group_y       = groups_from_threads(thread_group_y, cs_regs.cs_regs.num_thread_y);
		thread_group_z       = groups_from_threads(thread_group_z, cs_regs.cs_regs.num_thread_z);

		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: use-thread-dimensions %ux%ux%u / %ux%ux%u -> "
			     "groups %ux%ux%u\n",
			     old_x, old_y, old_z, std::max(cs_regs.cs_regs.num_thread_x, 1u),
			     std::max(cs_regs.cs_regs.num_thread_y, 1u),
			     std::max(cs_regs.cs_regs.num_thread_z, 1u), thread_group_x, thread_group_y,
			     thread_group_z);
		}
	}

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	buffer.EndRendering();
	auto& pipeline =
	    m_context.GetPipelineCache().CreateComputePipeline(input_info, sh_ctx.GetCs(), cs_shader);
	auto bindings = PrepareBindings(buffer, input_info.stage, vk::ShaderStageFlagBits::eCompute,
	                                DescriptorCache::Stage::Compute);
	RebindBuffers(buffer, bindings);
	RebindImages(buffer, bindings);

	auto vk_buffer = buffer.Handle();
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline.pipeline_layout, bindings);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);

	bool has_storage_writes = HasShaderBufferWrites(input_info.stage);
	has_storage_writes =
	    std::any_of(program.info.images.begin(), program.info.images.end(),
	                [](const auto& image) {
		                return image.written &&
		                       (image.kind == ShaderRecompiler::IR::ResourceKind::StorageImage ||
		                        image.kind == ShaderRecompiler::IR::ResourceKind::StorageImageUint);
	                }) ||
	    has_storage_writes;
	if (has_storage_writes) {
		ShaderWriteBarrier(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
	ResetBindings();
}

} // namespace Libs::Graphics
