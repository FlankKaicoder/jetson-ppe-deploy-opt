#include "trt_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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
    for (const auto* required : {"--engine", "--input", "--output", "--summary"}) {
        if (args.find(required) == args.end()) {
            throw std::runtime_error(std::string("missing argument: ") + required);
        }
    }
    return args;
}

std::vector<float> read_floats(const std::string& path, std::size_t elements) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open input tensor: " + path);
    }
    const auto bytes = static_cast<std::size_t>(stream.tellg());
    if (bytes != elements * sizeof(float)) {
        throw std::runtime_error("input tensor byte count mismatch");
    }
    std::vector<float> values(elements);
    stream.seekg(0, std::ios::beg);
    if (!stream.read(
            reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("cannot read complete input tensor");
    }
    return values;
}

void write_floats(const std::string& path, const std::vector<float>& values) {
    std::ofstream stream(path, std::ios::binary);
    if (!stream || !stream.write(
            reinterpret_cast<const char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)))) {
        throw std::runtime_error("cannot write output tensor: " + path);
    }
}

void write_stats(
    std::ostream& stream,
    const std::string& prefix,
    const ppe::TimingStats& stats) {
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
        ppe::TrtRuntime runtime(args.at("--engine"));
        const auto input = read_floats(
            args.at("--input"), runtime.input_info().elements);
        const auto result = runtime.infer(input, warmup, iterations);
        const bool finite = std::all_of(
            result.output.begin(), result.output.end(),
            [](float value) { return std::isfinite(value); });
        if (!finite) {
            throw std::runtime_error("output contains NaN or Inf");
        }
        write_floats(args.at("--output"), result.output);
        std::ofstream summary(args.at("--summary"));
        if (!summary) {
            throw std::runtime_error("cannot open summary output");
        }
        summary << std::setprecision(10)
                << "result=PASS\n"
                << "input_name=" << runtime.input_info().name << '\n'
                << "input_elements=" << runtime.input_info().elements << '\n'
                << "input_bytes=" << runtime.input_info().bytes << '\n'
                << "output_name=" << runtime.output_info().name << '\n'
                << "output_elements=" << runtime.output_info().elements << '\n'
                << "output_bytes=" << runtime.output_info().bytes << '\n'
                << "output_finite=true\n"
                << "warmup_iterations=" << warmup << '\n'
                << "timed_iterations=" << iterations << '\n'
                << "timing_scope=H2D_ENQUEUE_D2H_SYNCHRONIZE\n";
        write_stats(summary, "host_total", result.host_total);
        write_stats(summary, "cuda_total", result.cuda_total);
        std::cout << std::setprecision(6)
                  << "result=PASS output_elements=" << result.output.size()
                  << " host_mean_ms=" << result.host_total.mean_ms
                  << " cuda_mean_ms=" << result.cuda_total.mean_ms << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
