#include "cuda_smoke.h"

#include <NvInfer.h>
#include <NvInferVersion.h>
#include <gst/gst.h>
#include <opencv2/core.hpp>

#include <iostream>
#include <memory>

namespace
{

class TrtLogger final : public nvinfer1::ILogger
{
public:
    void log(
        Severity severity,
        const char* message) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cerr
                << "[TensorRT] "
                << (message != nullptr ? message : "")
                << '\n';
        }
    }
};

template <typename T>
struct TrtDeleter
{
    void operator()(T* object) const noexcept
    {
        delete object;
    }
};

}  // namespace

int main()
{
    bool overall_ok = true;

    std::cout << "========== CUDA device and kernel ==========\n";

    CudaSmokeResult cuda_result;
    const bool cuda_ok = runCudaSmokeTest(cuda_result);

    std::cout
        << "cuda_device_count="
        << cuda_result.device_count
        << '\n';

    std::cout
        << "cuda_device_name="
        << cuda_result.device_name
        << '\n';

    std::cout
        << "cuda_compute_capability="
        << cuda_result.compute_major
        << '.'
        << cuda_result.compute_minor
        << '\n';

    std::cout
        << "cuda_global_memory_bytes="
        << cuda_result.global_memory_bytes
        << '\n';

    std::cout
        << "cuda_kernel_max_abs_error="
        << cuda_result.max_abs_error
        << '\n';

    std::cout
        << "cuda_kernel_test="
        << (cuda_ok ? "PASS" : "FAIL")
        << '\n';

    overall_ok &= cuda_ok;

    std::cout << "\n========== TensorRT ==========\n";

    std::cout
        << "tensorrt_compile_version="
        << NV_TENSORRT_MAJOR
        << '.'
        << NV_TENSORRT_MINOR
        << '.'
        << NV_TENSORRT_PATCH
        << '\n';

    TrtLogger logger;

    std::unique_ptr<
        nvinfer1::IRuntime,
        TrtDeleter<nvinfer1::IRuntime>>
        runtime(nvinfer1::createInferRuntime(logger));

    const bool tensorrt_ok =
        runtime != nullptr;

    std::cout
        << "tensorrt_runtime_create="
        << (tensorrt_ok ? "PASS" : "FAIL")
        << '\n';

    overall_ok &= tensorrt_ok;

    std::cout << "\n========== OpenCV ==========\n";

    cv::Mat test_image(
        8,
        8,
        CV_8UC3,
        cv::Scalar(10, 20, 30));

    const bool opencv_ok =
        !test_image.empty() &&
        test_image.rows == 8 &&
        test_image.cols == 8 &&
        test_image.channels() == 3;

    std::cout
        << "opencv_version="
        << CV_VERSION
        << '\n';

    std::cout
        << "opencv_mat_test="
        << (opencv_ok ? "PASS" : "FAIL")
        << '\n';

    overall_ok &= opencv_ok;

    std::cout << "\n========== GStreamer ==========\n";

    gst_init(nullptr, nullptr);

    guint gst_major = 0;
    guint gst_minor = 0;
    guint gst_micro = 0;
    guint gst_nano = 0;

    gst_version(
        &gst_major,
        &gst_minor,
        &gst_micro,
        &gst_nano);

    const bool gstreamer_ok =
        gst_major >= 1;

    std::cout
        << "gstreamer_runtime_version="
        << gst_major
        << '.'
        << gst_minor
        << '.'
        << gst_micro
        << '\n';

    std::cout
        << "gstreamer_init_test="
        << (gstreamer_ok ? "PASS" : "FAIL")
        << '\n';

    overall_ok &= gstreamer_ok;

    std::cout << "\n========== final ==========\n";

    std::cout
        << "overall="
        << (overall_ok ? "PASS" : "FAIL")
        << '\n';

    return overall_ok ? 0 : 1;
}
