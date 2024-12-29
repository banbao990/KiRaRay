#pragma once
#include <cuda_runtime.h>
#include <optix.h>

#include "common.h"
#include "device/buffer.h"

#include "renderpass.h"
#include "window.h"

NAMESPACE_BEGIN(krr)

class DenoiseBackend {
public:
	enum class PixelFormat {
		FLOAT3,
		FLOAT4
	};

	DenoiseBackend() = default;

	void initialize();

	void denoise(CUstream stream, float *rgb, float *normal, float *albedo, float *result);

	void resize(Vector2i size);

	void setHaveGeometry(bool haveGeometry);
	bool haveGeometry() const { return haveGeometryBuffer; }

	void setPixelFormat(PixelFormat format);
	void setProps(bool haveGeometry, PixelFormat format);

private:
	Vector2i resolution;
	PixelFormat pixelFormat{PixelFormat::FLOAT4};
	bool haveGeometryBuffer{}, initialized{};
	OptixDenoiser denoiserHandle{};
	OptixDenoiserSizes memorySizes;
	CUDABuffer denoiserState, scratchBuffer, intensity;
};

class DenoisePass : public RenderPass {
public:
	using RenderPass::RenderPass;
	using SharedPtr = std::shared_ptr<DenoisePass>;
	KRR_REGISTER_PASS_DEC(DenoisePass);

	void render(RenderContext *context) override;
	void renderUI() override;
	void resize(const Vector2i &size) override;
	string getName() const override { return "DenoisePass"; }

	void denoise(float *rgb, float *result, DenoiseBackend::PixelFormat pixelFormat, float *normal,
				 float *albedo);

	friend void from_json(const json &j, DenoisePass &p) {
		p.mUseGeometry					= j.value("useGeometry", false);
		p.mPrepareGeometryBufferOutside = j.value("prepareGeometryBufferOutside", false);
	}

	friend void to_json(json &j, const DenoisePass &p) {
		j.update({{"useGeometry", p.mUseGeometry}});
	}

public:
	constexpr static char CTX_JSON_GBUFFER[] = "DENOISE_GBUFFER";

private:
	bool mPrepareGeometryBufferOutside{false};
	bool mUseGeometry{};
	TypedBuffer<RGBA> mColorBuffer;
	DenoiseBackend mBackend;
};

NAMESPACE_END(krr)