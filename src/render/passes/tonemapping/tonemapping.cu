#include "tonemapping.h"

#include "device/cuda.h"
#include "device/context.h"
#include "util/math_utils.h"
#include "render/profiler/profiler.h"

NAMESPACE_BEGIN(krr)

KRR_CALLABLE RGB toneMapAces(RGB color) {
	color *= 0.6;
	float A = 2.51;
	float B = 0.03;
	float C = 2.43;
	float D = 0.59;
	float E = 0.14;
	color	= clamp((color * (A * color + B)) / (color * (C * color + D) + E), 0, 1);
	return color;
}

KRR_CALLABLE RGB toneMapReinhard(RGB color) {
	float lum	   = luminance(color);
	float reinhard = lum / (lum + 1);
	return RGB(color * reinhard).safeDiv(lum);
}

KRR_CALLABLE RGB toneMapUC2(RGB color) {
	float A = 0.22; // Shoulder Strength
	float B = 0.3;	// Linear Strength
	float C = 0.1;	// Linear Angle
	float D = 0.2;	// Toe Strength
	float E = 0.01; // Toe Numerator
	float F = 0.3;	// Toe Denominator
	color	= ((color * (A * color + C * B) + D * E) / (color * (A * color + B) + D * F)) - (E / F);
	return color;
}

KRR_CALLABLE RGB toneMapHejiHableAlu(RGB color) {
	color = (color - 0.004f).cwiseMax(0);
	color = (color * (6.2f * color + 0.5f)) / (color * (6.2f * color + 1.7f) + 0.06f);
	// Result includes sRGB conversion
	return color.pow(2.2);
}

KRR_CALLABLE float linearToSrgb(float value) {
	// https://github.com/pgrit/SimpleImageIO/blob/ee563313bee9ea7db18e67c2b76ace3a34d83623/Core/manipulation.cpp#L6
	if (value > 0.0031308) {
		return 1.055f * (std::pow(value, (1.0f / 2.4f))) - 0.055f;
	} else {
		return 12.92f * value;
	}
}

KRR_CALLABLE RGB linearToSrgb(RGB color) {
	RGB result;
	for (int i = 0; i < RGB::dim; ++i) {
		result[i] = linearToSrgb(color[i]);
	}
	return result;
}

KRR_DEVICE_FUNCTION RGB colorJetMap(float vIn, float vMax) {
	RGB out = RGB(1.f, 1.f, 1.f);
	float v = vIn;
	if (v < (0.25 * vMax)) {
		out[0] = 0;
		out[1] = 4 * v / vMax;
	} else if (v < (0.5 * vMax)) {
		out[0] = 0;
		out[2] = 1 + 4 * (0.25 * vMax - v) / vMax;
	} else if (v < (0.75 * vMax)) {
		out[0] = 4 * (v - 0.5 * vMax) / vMax;
		out[2] = 0;
	} else {
		v	   = min(v, vMax);
		out[1] = 1 + 4 * (0.75 * vMax - v) / vMax;
		out[2] = 0;
	}
	return out;
}

void ToneMappingPass::renderUI() {
	static const char *operators[] = {"Linear",	   "Reinhard", "Aces", "Uncharted2",
									  "HejiHable", "Lin2Srgb", "Jet"};
	ui::Checkbox("Enabled", &mEnable);
	if (mEnable) {
		ui::DragFloat("Exposure compensation", &mExposureCompensation, 0.01, 0.01, 100, "%.2f");
		ui::Combo("Tonemap operator", (int *) &mOperator, operators, (int) Operator::NumsOperators);
		ui::Checkbox("Use Gamma", &mUseGamma);

		if (mOperator == Operator::Jet) {
			ui::SliderFloat("Show Scalar Jet UpBound[log2]", &mJetMaxUpBound, 0.0f, 20.0f);
			const float jetMaxExp = pow(2, mJetMaxUpBound);
			ui::SliderFloat("Show Scalar Jet Max", &mJetMax, 0.1f, jetMaxExp);
			mJetMax = clamp(mJetMax, 0.0001f, jetMaxExp);
			ui::Checkbox("Show Tint", &mJetShowTint);
		}
	}
}

void ToneMappingPass::render(RenderContext *context) {
	PROFILE("Tong mapping pass");
	CUstream &stream			 = KRR_DEFAULT_STREAM;
	RGB colorTransform			 = RGB(mExposureCompensation);
	CudaRenderTarget frameBuffer = context->getColorTexture()->getCudaRenderTarget();

	const auto frameSize = getFrameSize();
	const int width = frameSize[0], height = frameSize[1];
	GPUParallelFor(
		width * height,
		KRR_DEVICE_LAMBDA(int pixelId) {
			RGB color = frameBuffer.read(pixelId).head<3>() * colorTransform;
			float v	  = 0.0f;

			switch (mOperator) {
				case krr::ToneMappingPass::Operator::Linear:
					break;
				case krr::ToneMappingPass::Operator::Reinhard:
					color = toneMapReinhard(color);
					break;
				case krr::ToneMappingPass::Operator::Aces:
					color = toneMapAces(color);
					break;
				case krr::ToneMappingPass::Operator::Uncharted2:
					color = toneMapUC2(color);
					break;
				case krr::ToneMappingPass::Operator::HejiHable:
					color = toneMapHejiHableAlu(color);
					break;
				case krr::ToneMappingPass::Operator::Lin2Srgb:
					color = linearToSrgb(color);
					break;
				case krr::ToneMappingPass::Operator::Jet:
					v = color.mean();
					if (mJetShowTint) {
						v = float(pixelId % width) / width;
						v *= mJetMax; // Scale to [0, mJetMax]
					}
					color = colorJetMap(v, mJetMax);
				default:
					break;
			}
			if (mUseGamma) color = color.pow(0.45454545f);
			frameBuffer.write(RGBA(color, 1.f), pixelId);
		},
		KRR_DEFAULT_STREAM);
}

KRR_REGISTER_PASS_DEF(ToneMappingPass);
NAMESPACE_END(krr)