#include "denoise.h"
#include "util/check.h"
#include "render/profiler/profiler.h"
#include "device/context.h"
#include "device/cuda.h"

NAMESPACE_BEGIN(krr)

void DenoiseBackend::initialize() {
	CUcontext cudaContext;
	cuCtxGetCurrent(&cudaContext);
	CHECK(cudaContext != nullptr);
	const OptixDeviceContext &optixContext = gpContext->optixContext;
	if (mDenoiserHandle) optixDenoiserDestroy(mDenoiserHandle);

	OptixDenoiserOptions options = {};
#if (OPTIX_VERSION >= 70300)
	if (mHaveGeometryBuffer) options.guideAlbedo = options.guideNormal = 1;

	OPTIX_CHECK(optixDenoiserCreate(optixContext, OPTIX_DENOISER_MODEL_KIND_HDR, &options,
									&mDenoiserHandle));
#else
	options.inputKind =
		haveGeometryBuffer
			? (pixelFormat == PixelFormat::FLOAT3 ? OPTIX_DENOISER_INPUT_Color3f_ALBEDO_NORMAL
												  : OPTIX_DENOISER_INPUT_Color4f_ALBEDO_NORMAL)
			: (pixelFormat == PixelFormat::FLOAT3 ? OPTIX_DENOISER_INPUT_Color3f
												  : OPTIX_DENOISER_INPUT_Color4f);

	OPTIX_CHECK(optixDenoiserCreate(optixContext, &options, &denoiserHandle));

	OPTIX_CHECK(optixDenoiserSetModel(denoiserHandle, OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0));
#endif
	// re-create compute memory resources.
	OPTIX_CHECK(optixDenoiserComputeMemoryResources(mDenoiserHandle, mResolution[0], mResolution[1],
													&mMemorySizes));

	mDenoiserState.resize(mMemorySizes.stateSizeInBytes);
	mScratchBuffer.resize(mMemorySizes.withoutOverlapScratchSizeInBytes);
	mIntensity.resize(sizeof(float));

	OPTIX_CHECK(optixDenoiserSetup(
		mDenoiserHandle, KRR_DEFAULT_STREAM, mResolution[0], mResolution[1],
		CUdeviceptr(mDenoiserState.data()), mMemorySizes.stateSizeInBytes,
		CUdeviceptr(mScratchBuffer.data()), mMemorySizes.withoutOverlapScratchSizeInBytes));
}

void DenoiseBackend::denoise(CUstream stream, float *rgb, float *normal, float *albedo,
							 float *result) {

	// don't support FLOAT1
	const int strideTypes[]					  = {3, 4};
	const OptixPixelFormat pixelFormatTypes[] = {OPTIX_PIXEL_FORMAT_FLOAT3,
												 OPTIX_PIXEL_FORMAT_FLOAT4};
	const int stride						  = strideTypes[(uint) mPixelFormat] * sizeof(float);
	const OptixPixelFormat pixelFormat		  = pixelFormatTypes[(uint) mPixelFormat];

	std::array<OptixImage2D, 3> inputLayers = {};
	const int inputPixelStride				= stride;
	const int outputPixelStride				= stride;
	const int nLayers						= mHaveGeometryBuffer ? 3 : 1;

	for (int i = 0; i < nLayers; ++i) {
		inputLayers[i].width			  = mResolution[0];
		inputLayers[i].height			  = mResolution[1];
		inputLayers[i].rowStrideInBytes	  = mResolution[0] * inputPixelStride;
		inputLayers[i].pixelStrideInBytes = inputPixelStride;
		inputLayers[i].format			  = pixelFormat;
	}

	inputLayers[0].data = CUdeviceptr(rgb);
	if (mHaveGeometryBuffer) {
		// normal and albedo is in float3 format.
		inputLayers[1].format			  = OPTIX_PIXEL_FORMAT_FLOAT3;
		inputLayers[1].data				  = CUdeviceptr(albedo);
		inputLayers[1].rowStrideInBytes	  = mResolution[0] * sizeof(RGB);
		inputLayers[1].pixelStrideInBytes = sizeof(RGB);
		inputLayers[2].format			  = OPTIX_PIXEL_FORMAT_FLOAT3;
		inputLayers[2].data				  = CUdeviceptr(normal);
		inputLayers[2].pixelStrideInBytes = sizeof(RGB);
	} else {
		CHECK(normal == nullptr && albedo == nullptr);
	}

	OptixImage2D outputImage	   = {};
	outputImage.width			   = mResolution[0];
	outputImage.height			   = mResolution[1];
	outputImage.rowStrideInBytes   = mResolution[0] * outputPixelStride;
	outputImage.pixelStrideInBytes = outputPixelStride;
	outputImage.format			   = pixelFormat;
	outputImage.data			   = CUdeviceptr(result);

	OPTIX_CHECK(optixDenoiserComputeIntensity(
		mDenoiserHandle, stream, &inputLayers[0], CUdeviceptr(mIntensity.data()),
		CUdeviceptr(mScratchBuffer.data()), mMemorySizes.withoutOverlapScratchSizeInBytes));

	OptixDenoiserParams params = {};
#if (OPTIX_VERSION < 80000 && OPTIX_VERSION >= 70500)
	params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
#elif (OPTIX_VERSION < 70500)
	params.denoiseAlpha = 0;
#endif
	params.hdrIntensity = CUdeviceptr(mIntensity.data());
	params.blendFactor	= 0;

#if (OPTIX_VERSION >= 70300)
	OptixDenoiserGuideLayer guideLayer = {};
	if (mHaveGeometryBuffer) {
		guideLayer.albedo = inputLayers[1];
		guideLayer.normal = inputLayers[2];
	}

	OptixDenoiserLayer layers = {};
	layers.input			  = inputLayers[0];
	layers.output			  = outputImage;
	OPTIX_CHECK(optixDenoiserInvoke(
		mDenoiserHandle, stream /* stream */, &params, CUdeviceptr(mDenoiserState.data()),
		mMemorySizes.stateSizeInBytes, &guideLayer, &layers,
		1 /* # layers to denoise(layers.size) */, 0 /* offset x */, 0 /* offset y */,
		CUdeviceptr(mScratchBuffer.data()), mMemorySizes.withoutOverlapScratchSizeInBytes));
#else
	OPTIX_CHECK(optixDenoiserInvoke(denoiserHandle, stream /* stream */, &params,
									CUdeviceptr(denoiserState.data()), memorySizes.stateSizeInBytes,
									inputLayers.data(), 1, 0 /* offset x */, 0 /* offset y */,
									&outputImage, CUdeviceptr(scratchBuffer.data()),
									memorySizes.withoutOverlapScratchSizeInBytes));
#endif
}

void DenoiseBackend::resize(Vector2i size) {
	if (mResolution == size) {
		return;
	}
	mResolution = size;
	initialize();
}

void DenoiseBackend::setHaveGeometry(bool haveGeometry) {
	if (mHaveGeometryBuffer == haveGeometry) {
		return;
	}
	mHaveGeometryBuffer = haveGeometry;
	initialize();
}

void DenoiseBackend::setProps(bool haveGeometry, PixelFormat format) {
	bool changed = false;
	if (mPixelFormat != format) {
		mPixelFormat = format;
		changed		 = true;
	}

	if (mHaveGeometryBuffer != haveGeometry) {
		mHaveGeometryBuffer = haveGeometry;
		changed				= true;
	}

	if (changed) {
		initialize();
	}
}

void DenoiseBackend::setPixelFormat(PixelFormat format) {
	if (mPixelFormat == format) {
		return;
	}
	mPixelFormat = format;
	initialize();
}

void DenoisePass::render(RenderContext *context) {
	PROFILE("Denoise");

	// TODO: The general case of denoising guided by geometry features is not implemented yet TaT,
	// so should set the flag to false.
	mBackend.setProps(false, DenoiseBackend::PixelFormat::FLOAT4);

	const auto size			   = context->getRenderTarget()->getSize();
	CudaRenderTarget cudaFrame = context->getColorTexture()->getCudaRenderTarget();
	RGBA *colorBuffer		   = mColorBuffer.data();
	GPUParallelFor(
		size[0] * size[1],
		[=] KRR_DEVICE(int pixelId) mutable { colorBuffer[pixelId] = cudaFrame.read(pixelId); },
		KRR_DEFAULT_STREAM);
	mBackend.denoise(KRR_DEFAULT_STREAM, (float *) colorBuffer, nullptr, nullptr,
					 (float *) colorBuffer);
	GPUParallelFor(
		size[0] * size[1],
		[=] KRR_DEVICE(int pixelId) mutable { cudaFrame.write(colorBuffer[pixelId], pixelId); },
		KRR_DEFAULT_STREAM);
}

void DenoisePass::renderUI() {
	ui::Checkbox("Enabled", &mEnable);
	if (!mEnable) {
		return;
	}
	static const char *sPixelFormats[] = {"FLOAT3", "FLOAT4"};
	ui::Text("Pixel Format: %s", sPixelFormats[(uint) mBackend.getPixelFormat()]);
	ui::Checkbox("Use geometry buffer", &mUseGeometry);
	if (mUseGeometry) {
		if (mPrepareGeometryBufferOutside) {
			ui::SameLine();
			ui::Text("[You should prepare geometry buffer outside]");
		} else {
			Log(Fatal, "Denoising guided by geometry features is not implemented yet TaT");
		}
		mBackend.setHaveGeometry(mUseGeometry);
	}
}

void DenoisePass::resize(const Vector2i &size) {
	RenderPass::resize(size);
	mBackend.resize(size);
	mColorBuffer.resize(size[0] * size[1]);
}

void DenoisePass::denoise(float *rgb, float *result, DenoiseBackend::PixelFormat pixelFormat,
						  float *normal, float *albedo) {
	if (mUseGeometry) {
		if (normal == nullptr || albedo == nullptr) {
			Log(Fatal, "Normal and albedo should be provided when using geometry buffer.");
		}
	} else {
		normal = albedo = nullptr;
	}

	mBackend.setProps(mUseGeometry, pixelFormat);
	mBackend.denoise(KRR_DEFAULT_STREAM, rgb, normal, albedo, result);
}

KRR_REGISTER_PASS_DEF(DenoisePass);
NAMESPACE_END(krr)
