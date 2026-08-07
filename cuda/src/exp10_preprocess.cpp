#include "cuda_preprocess.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::map<std::string, std::string> parse_args(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc || std::string(argv[index]).rfind("--", 0) != 0) {
            throw std::runtime_error("arguments must be --key value pairs");
        }
        args[argv[index]] = argv[index + 1];
    }
    for (const auto* required :
         {"--image", "--cpu-output", "--cuda-output", "--summary"}) {
        if (args.find(required) == args.end()) {
            throw std::runtime_error(std::string("missing argument: ") + required);
        }
    }
    return args;
}

double percentile(const std::vector<double>& sorted, double quantile) {
    const double position = quantile * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}

ppe::PreprocessTimingStats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    ppe::PreprocessTimingStats result;
    result.mean_ms =
        std::accumulate(values.begin(), values.end(), 0.0) /
        static_cast<double>(values.size());
    result.p50_ms = percentile(values, 0.50);
    result.p95_ms = percentile(values, 0.95);
    result.p99_ms = percentile(values, 0.99);
    result.min_ms = values.front();
    result.max_ms = values.back();
    return result;
}

std::vector<float> cpu_preprocess(
    const cv::Mat& image,
    const ppe::LetterboxGeometry& geometry) {
    cv::Mat resized;
    if (image.cols == geometry.resized_width &&
        image.rows == geometry.resized_height) {
        resized = image.clone();
    } else {
        cv::resize(
            image, resized,
            cv::Size(geometry.resized_width, geometry.resized_height),
            0.0, 0.0, cv::INTER_LINEAR);
    }
    cv::Mat letterboxed;
    cv::copyMakeBorder(
        resized, letterboxed,
        geometry.padding_top, geometry.padding_bottom,
        geometry.padding_left, geometry.padding_right,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    if (letterboxed.rows != geometry.target_size ||
        letterboxed.cols != geometry.target_size ||
        letterboxed.type() != CV_8UC3) {
        throw std::runtime_error("invalid CPU letterbox output");
    }
    const int plane = geometry.target_size * geometry.target_size;
    std::vector<float> output(static_cast<std::size_t>(plane) * 3);
    for (int y = 0; y < geometry.target_size; ++y) {
        const auto* row = letterboxed.ptr<std::uint8_t>(y);
        for (int x = 0; x < geometry.target_size; ++x) {
            const int pixel = y * geometry.target_size + x;
            output[pixel] = static_cast<float>(row[x * 3 + 2]) / 255.0F;
            output[plane + pixel] =
                static_cast<float>(row[x * 3 + 1]) / 255.0F;
            output[2 * plane + pixel] =
                static_cast<float>(row[x * 3]) / 255.0F;
        }
    }
    return output;
}

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("cannot write tensor: " + path);
    }
}

void write_stats(
    std::ostream& stream,
    const std::string& prefix,
    const ppe::PreprocessTimingStats& stats) {
    stream << prefix << "_mean_ms=" << stats.mean_ms << '\n'
           << prefix << "_p50_ms=" << stats.p50_ms << '\n'
           << prefix << "_p95_ms=" << stats.p95_ms << '\n'
           << prefix << "_p99_ms=" << stats.p99_ms << '\n'
           << prefix << "_min_ms=" << stats.min_ms << '\n'
           << prefix << "_max_ms=" << stats.max_ms << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto args = parse_args(argc, argv);
        const int warmup = args.count("--warmup") ? std::stoi(args.at("--warmup")) : 2;
        const int iterations =
            args.count("--iterations") ? std::stoi(args.at("--iterations")) : 5;
        const cv::Mat image = cv::imread(args.at("--image"), cv::IMREAD_COLOR);
        if (image.empty() || !image.isContinuous() || image.type() != CV_8UC3) {
            throw std::runtime_error("invalid or non-contiguous BGR image");
        }
        const auto geometry = ppe::make_letterbox_geometry(
            image.cols, image.rows, 640);

        std::vector<float> cpu_output;
        for (int index = 0; index < warmup; ++index) {
            cpu_output = cpu_preprocess(image, geometry);
        }
        std::vector<double> cpu_values;
        cpu_values.reserve(static_cast<std::size_t>(iterations));
        for (int index = 0; index < iterations; ++index) {
            const auto start = std::chrono::steady_clock::now();
            cpu_output = cpu_preprocess(image, geometry);
            const auto end = std::chrono::steady_clock::now();
            cpu_values.push_back(std::chrono::duration<double, std::milli>(
                end - start).count());
        }
        const auto cpu_stats = summarize(cpu_values);
        const auto cuda_result = ppe::run_cuda_preprocess(
            image.ptr<std::uint8_t>(), geometry, warmup, iterations);
        const bool finite = std::all_of(
            cuda_result.output.begin(), cuda_result.output.end(),
            [](float value) { return std::isfinite(value); });
        if (!finite) {
            throw std::runtime_error("CUDA output contains NaN or Inf");
        }
        write_floats(args.at("--cpu-output"), cpu_output);
        write_floats(args.at("--cuda-output"), cuda_result.output);

        std::ofstream summary(args.at("--summary"));
        if (!summary) {
            throw std::runtime_error("cannot open summary path");
        }
        summary << std::setprecision(10)
                << "result=PASS\n"
                << "source_width=" << geometry.source_width << '\n'
                << "source_height=" << geometry.source_height << '\n'
                << "target_size=" << geometry.target_size << '\n'
                << "resized_width=" << geometry.resized_width << '\n'
                << "resized_height=" << geometry.resized_height << '\n'
                << "padding_left=" << geometry.padding_left << '\n'
                << "padding_right=" << geometry.padding_right << '\n'
                << "padding_top=" << geometry.padding_top << '\n'
                << "padding_bottom=" << geometry.padding_bottom << '\n'
                << "ratio=" << geometry.ratio << '\n'
                << "output_shape=1,3,640,640\n"
                << "output_finite=true\n"
                << "warmup_iterations=" << warmup << '\n'
                << "timed_iterations=" << iterations << '\n';
        write_stats(summary, "cpu", cpu_stats);
        write_stats(summary, "cuda_kernel", cuda_result.kernel_only);
        write_stats(summary, "cuda_total", cuda_result.total_with_transfers);
        std::cout << std::setprecision(6)
                  << "result=PASS source=" << image.rows << 'x' << image.cols
                  << " cpu_mean_ms=" << cpu_stats.mean_ms
                  << " kernel_mean_ms=" << cuda_result.kernel_only.mean_ms
                  << " total_mean_ms=" << cuda_result.total_with_transfers.mean_ms
                  << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
