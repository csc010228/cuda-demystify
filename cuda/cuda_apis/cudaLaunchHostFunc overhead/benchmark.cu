#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t error_ = (call);                                                 \
    if (error_ != cudaSuccess) {                                                 \
      std::ostringstream message_;                                              \
      message_ << #call << " failed at " << __FILE__ << ':' << __LINE__ << ": " \
               << cudaGetErrorString(error_);                                   \
      throw std::runtime_error(message_.str());                                 \
    }                                                                           \
  } while (0)

struct Options {
  int device = 0;
  int kernels = 16;
  std::vector<double> kernel_ci{16.0};
  std::size_t elements = 1u << 20;
  int threads = 256;
  int callback_every = 1;
  double callback_work_us = 0.0;
  int warmups = 3;
  int trials = 20;
  int cpu_load_threads = 0;
  int gpu_load_streams = 0;
  int gpu_load_kernels = 100;
  double gpu_load_ci = 64.0;
  bool csv = false;
  bool kernels_explicit = false;
  bool ci_list_explicit = false;
};

struct CallbackData {
  std::uint64_t api_return_ns = 0;
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  std::uint64_t work_ns = 0;
};

struct RunResult {
  double stream_ms = 0.0;
  double submit_us = 0.0;
  std::vector<double> api_ns;
  std::vector<double> queue_us;
  std::vector<double> callback_us;
};

struct Stats {
  double mean = 0.0;
  double median = 0.0;
  double p95 = 0.0;
  double stdev = 0.0;
};

static std::uint64_t now_ns() {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          Clock::now().time_since_epoch())
          .count());
}

static Stats stats(std::vector<double> values) {
  Stats result;
  if (values.empty()) return result;
  result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double variance = 0.0;
  for (double value : values) variance += (value - result.mean) * (value - result.mean);
  result.stdev = std::sqrt(variance / values.size());
  std::sort(values.begin(), values.end());
  auto percentile = [&](double p) {
    const double index = p * static_cast<double>(values.size() - 1);
    const std::size_t low = static_cast<std::size_t>(index);
    const std::size_t high = std::min(low + 1, values.size() - 1);
    const double fraction = index - static_cast<double>(low);
    return values[low] + fraction * (values[high] - values[low]);
  };
  result.median = percentile(0.50);
  result.p95 = percentile(0.95);
  return result;
}

static std::vector<double> parse_list(const std::string& text) {
  std::vector<double> result;
  std::stringstream input(text);
  std::string item;
  while (std::getline(input, item, ',')) {
    if (item.empty()) throw std::invalid_argument("empty item in intensity list");
    std::size_t consumed = 0;
    double value = std::stod(item, &consumed);
    if (consumed != item.size() || value < 0.0 || !std::isfinite(value))
      throw std::invalid_argument("invalid computational intensity: " + item);
    result.push_back(value);
  }
  if (result.empty()) throw std::invalid_argument("intensity list cannot be empty");
  return result;
}

static long long parse_integer(const char* option, const char* value) {
  std::size_t consumed = 0;
  long long result = std::stoll(value, &consumed);
  if (consumed != std::string(value).size())
    throw std::invalid_argument(std::string("invalid value for ") + option + ": " + value);
  return result;
}

static int parse_int(const char* option, const char* value) {
  const long long result = parse_integer(option, value);
  if (result < std::numeric_limits<int>::min() ||
      result > std::numeric_limits<int>::max())
    throw std::invalid_argument(std::string("value for ") + option + " is out of range");
  return static_cast<int>(result);
}

static std::size_t parse_size(const char* option, const char* value) {
  const long long result = parse_integer(option, value);
  if (result <= 0)
    throw std::invalid_argument(std::string("value for ") + option + " must be positive");
  return static_cast<std::size_t>(result);
}

static double parse_double(const char* option, const char* value) {
  std::size_t consumed = 0;
  double result = std::stod(value, &consumed);
  if (consumed != std::string(value).size() || !std::isfinite(result))
    throw std::invalid_argument(std::string("invalid value for ") + option + ": " + value);
  return result;
}

static void usage(const char* program) {
  std::cout
      << "Usage: " << program << " [options]\n\n"
      << "Main workload:\n"
      << "  --kernels N                 number of kernels (default: 16)\n"
      << "  --kernel-ci FLOP_PER_BYTE   nominal CI for every kernel (default: 16)\n"
      << "  --kernel-ci-list A,B,...    CI for each kernel; list length sets --kernels\n"
      << "  --elements N                float elements processed per kernel (default: 1048576)\n"
      << "  --threads N                 threads per block (default: 256)\n"
      << "  --callback-every N          enqueue callback after every N kernels (default: 1)\n"
      << "  --callback-work-us US       busy work inside callback (default: 0)\n\n"
      << "Controlled system load:\n"
      << "  --cpu-load-threads N        CPU busy-loop workers (default: 0)\n"
      << "  --gpu-load-streams N        competing nonblocking CUDA streams (default: 0)\n"
      << "  --gpu-load-kernels N        kernels queued per competing stream (default: 100)\n"
      << "  --gpu-load-ci FLOP_PER_BYTE CI of competing kernels (default: 64)\n\n"
      << "Measurement:\n"
      << "  --warmups N                 paired warmup iterations (default: 3)\n"
      << "  --trials N                  paired measured iterations (default: 20)\n"
      << "  --device N                  CUDA device ordinal (default: 0)\n"
      << "  --csv                       print one machine-readable result row\n"
      << "  --help                      show this help\n";
}

static Options parse_options(int argc, char** argv) {
  Options options;
  auto next = [&](int& i) -> const char* {
    if (++i >= argc) throw std::invalid_argument(std::string("missing value for ") + argv[i - 1]);
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else if (arg == "--kernels") {
      const char* option = argv[i];
      options.kernels = parse_int(option, next(i));
      options.kernels_explicit = true;
    } else if (arg == "--kernel-ci") {
      const char* option = argv[i];
      options.kernel_ci = {parse_double(option, next(i))};
      options.ci_list_explicit = false;
    } else if (arg == "--kernel-ci-list") {
      options.kernel_ci = parse_list(next(i));
      options.ci_list_explicit = true;
    } else if (arg == "--elements") {
      const char* option = argv[i];
      options.elements = parse_size(option, next(i));
    } else if (arg == "--threads") {
      const char* option = argv[i];
      options.threads = parse_int(option, next(i));
    } else if (arg == "--callback-every") {
      const char* option = argv[i];
      options.callback_every = parse_int(option, next(i));
    } else if (arg == "--callback-work-us") {
      const char* option = argv[i];
      options.callback_work_us = parse_double(option, next(i));
    } else if (arg == "--warmups") {
      const char* option = argv[i];
      options.warmups = parse_int(option, next(i));
    } else if (arg == "--trials") {
      const char* option = argv[i];
      options.trials = parse_int(option, next(i));
    } else if (arg == "--cpu-load-threads") {
      const char* option = argv[i];
      options.cpu_load_threads = parse_int(option, next(i));
    } else if (arg == "--gpu-load-streams") {
      const char* option = argv[i];
      options.gpu_load_streams = parse_int(option, next(i));
    } else if (arg == "--gpu-load-kernels") {
      const char* option = argv[i];
      options.gpu_load_kernels = parse_int(option, next(i));
    } else if (arg == "--gpu-load-ci") {
      const char* option = argv[i];
      options.gpu_load_ci = parse_double(option, next(i));
    } else if (arg == "--device") {
      const char* option = argv[i];
      options.device = parse_int(option, next(i));
    } else if (arg == "--csv") {
      options.csv = true;
    } else {
      throw std::invalid_argument("unknown option: " + arg);
    }
  }

  if (options.ci_list_explicit) {
    if (options.kernels_explicit && options.kernels != static_cast<int>(options.kernel_ci.size()))
      throw std::invalid_argument("--kernels must equal the --kernel-ci-list length");
    options.kernels = static_cast<int>(options.kernel_ci.size());
  } else {
    options.kernel_ci.resize(options.kernels, options.kernel_ci.front());
  }
  if (options.kernels <= 0 || options.elements == 0 || options.threads <= 0 ||
      options.threads > 1024 || options.callback_every <= 0 || options.warmups < 0 ||
      options.trials <= 0 || options.cpu_load_threads < 0 || options.gpu_load_streams < 0 ||
      options.gpu_load_kernels < 0 || options.callback_work_us < 0.0 ||
      options.gpu_load_ci < 0.0)
    throw std::invalid_argument("numeric options are outside their valid range");
  for (double ci : options.kernel_ci) {
    if (ci < 0.0 || !std::isfinite(ci))
      throw std::invalid_argument("kernel computational intensity must be finite and nonnegative");
  }
  return options;
}

// One load + one store = 8 bytes. Each loop iteration is one FMA = 2 FLOPs,
// so the requested nominal arithmetic intensity maps to ceil(4 * CI) iterations.
__global__ void compute_kernel(float* data, std::size_t count, int iterations,
                               unsigned int launch_id) {
  for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    float value = data[index] + static_cast<float>(launch_id & 7u) * 1.0e-7f;
#pragma unroll 1
    for (int iteration = 0; iteration < iterations; ++iteration)
      value = fmaf(value, 1.00000011920928955078125f, 1.1920928955078125e-7f);
    data[index] = value;
  }
}

static int ci_to_iterations(double ci) {
  const double iterations = std::ceil(ci * 4.0);
  if (iterations > static_cast<double>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("computational intensity is too large");
  return static_cast<int>(iterations);
}

static void CUDART_CB host_callback(void* raw) {
  auto* data = static_cast<CallbackData*>(raw);
  data->start_ns = now_ns();
  const std::uint64_t deadline = data->start_ns + data->work_ns;
  while (now_ns() < deadline) std::atomic_signal_fence(std::memory_order_seq_cst);
  data->end_ns = now_ns();
}

class CpuLoad {
 public:
  explicit CpuLoad(int count) {
    workers_.reserve(count);
    for (int i = 0; i < count; ++i) {
      workers_.emplace_back([this, i] {
        double value = 1.0 + i;
        while (!stop_.load(std::memory_order_relaxed)) {
          for (int j = 0; j < 10000; ++j) value = value * 1.0000001 + 0.0000001;
          sink_.fetch_add(static_cast<std::uint64_t>(value), std::memory_order_relaxed);
        }
      });
    }
  }
  ~CpuLoad() {
    stop_.store(true, std::memory_order_relaxed);
    for (auto& worker : workers_) worker.join();
  }

 private:
  std::atomic<bool> stop_{false};
  std::atomic<std::uint64_t> sink_{0};
  std::vector<std::thread> workers_;
};

class Benchmark {
 public:
  explicit Benchmark(const Options& options) : options_(options) {
    CUDA_CHECK(cudaSetDevice(options.device));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
    CUDA_CHECK(cudaEventCreateWithFlags(&load_gate_, cudaEventDisableTiming));
    CUDA_CHECK(cudaMalloc(&data_, options.elements * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(data_, 0, options.elements * sizeof(float), stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    for (int i = 0; i < options.gpu_load_streams; ++i) {
      cudaStream_t load_stream;
      float* load_data;
      CUDA_CHECK(cudaStreamCreateWithFlags(&load_stream, cudaStreamNonBlocking));
      CUDA_CHECK(cudaMalloc(&load_data, options.elements * sizeof(float)));
      CUDA_CHECK(cudaMemsetAsync(load_data, 0, options.elements * sizeof(float), load_stream));
      load_streams_.push_back(load_stream);
      load_data_.push_back(load_data);
    }
    for (cudaStream_t load_stream : load_streams_) CUDA_CHECK(cudaStreamSynchronize(load_stream));
  }

  ~Benchmark() {
    for (float* pointer : load_data_) cudaFree(pointer);
    for (cudaStream_t stream : load_streams_) cudaStreamDestroy(stream);
    cudaFree(data_);
    cudaEventDestroy(load_gate_);
    cudaEventDestroy(stop_);
    cudaEventDestroy(start_);
    cudaStreamDestroy(stream_);
  }

  RunResult run(bool with_callbacks) {
    RunResult result;
    const int callback_count = (options_.kernels + options_.callback_every - 1) /
                               options_.callback_every;
    std::vector<CallbackData> callbacks(with_callbacks ? callback_count : 0);
    for (CallbackData& callback : callbacks)
      callback.work_ns = static_cast<std::uint64_t>(options_.callback_work_us * 1000.0);

    launch_background();
    CUDA_CHECK(cudaEventRecord(start_, stream_));
    const auto submit_start = Clock::now();
    int callback_index = 0;
    for (int i = 0; i < options_.kernels; ++i) {
      launch_compute(stream_, data_, options_.kernel_ci[i], static_cast<unsigned int>(i));
      const bool callback_here = ((i + 1) % options_.callback_every == 0) ||
                                 (i + 1 == options_.kernels);
      if (with_callbacks && callback_here) {
        CallbackData& callback = callbacks[callback_index++];
        const auto api_start = Clock::now();
        CUDA_CHECK(cudaLaunchHostFunc(stream_, host_callback, &callback));
        const auto api_end = Clock::now();
        callback.api_return_ns = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                api_end.time_since_epoch())
                .count());
        result.api_ns.push_back(
            std::chrono::duration<double, std::nano>(api_end - api_start).count());
      }
    }
    const auto submit_end = Clock::now();
    CUDA_CHECK(cudaEventRecord(stop_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_));
    for (cudaStream_t load_stream : load_streams_) CUDA_CHECK(cudaStreamSynchronize(load_stream));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
    result.stream_ms = milliseconds;
    result.submit_us = std::chrono::duration<double, std::micro>(submit_end - submit_start).count();

    for (const CallbackData& callback : callbacks) {
      result.queue_us.push_back(static_cast<double>(callback.start_ns - callback.api_return_ns) / 1000.0);
      result.callback_us.push_back(static_cast<double>(callback.end_ns - callback.start_ns) / 1000.0);
    }
    return result;
  }

 private:
  void launch_compute(cudaStream_t stream, float* data, double ci, unsigned int id) {
    const int blocks = static_cast<int>(std::min<std::size_t>(
        (options_.elements + options_.threads - 1) / options_.threads, 65535));
    compute_kernel<<<blocks, options_.threads, 0, stream>>>(
        data, options_.elements, ci_to_iterations(ci), id);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  void launch_background() {
    if (load_streams_.empty() || options_.gpu_load_kernels == 0) return;
    CUDA_CHECK(cudaEventRecord(load_gate_, stream_));
    for (std::size_t s = 0; s < load_streams_.size(); ++s) {
      CUDA_CHECK(cudaStreamWaitEvent(load_streams_[s], load_gate_));
      for (int i = 0; i < options_.gpu_load_kernels; ++i)
        launch_compute(load_streams_[s], load_data_[s], options_.gpu_load_ci,
                       static_cast<unsigned int>(i));
    }
  }

  const Options& options_;
  cudaStream_t stream_{};
  cudaEvent_t start_{};
  cudaEvent_t stop_{};
  cudaEvent_t load_gate_{};
  float* data_{};
  std::vector<cudaStream_t> load_streams_;
  std::vector<float*> load_data_;
};

static double clock_floor_ns() {
  std::vector<double> samples;
  samples.reserve(10000);
  for (int i = 0; i < 10000; ++i) {
    const auto start = Clock::now();
    const auto stop = Clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(stop - start).count());
  }
  return stats(samples).median;
}

static std::string ci_description(const Options& options) {
  if (!options.ci_list_explicit) {
    std::ostringstream output;
    output << options.kernel_ci.front();
    return output.str();
  }
  std::ostringstream output;
  for (std::size_t i = 0; i < options.kernel_ci.size(); ++i) {
    if (i) output << ',';
    output << options.kernel_ci[i];
  }
  return output.str();
}

int main(int argc, char** argv) {
  try {
    Options options = parse_options(argc, argv);
    CUDA_CHECK(cudaSetDevice(options.device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device));
    int driver_version = 0;
    int runtime_version = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver_version));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    const int callback_count = (options.kernels + options.callback_every - 1) /
                               options.callback_every;
    const double timer_floor = clock_floor_ns();

    CpuLoad cpu_load(options.cpu_load_threads);
    Benchmark benchmark(options);
    for (int i = 0; i < options.warmups; ++i) {
      benchmark.run(false);
      benchmark.run(true);
    }

    std::vector<double> baseline_ms, callback_ms, device_overhead_us;
    std::vector<double> baseline_submit_us, callback_submit_us, submit_overhead_us;
    std::vector<double> api_ns, queue_us, callback_body_us;
    for (int trial = 0; trial < options.trials; ++trial) {
      RunResult baseline;
      RunResult callback;
      // Alternate order to reduce bias from thermal and frequency drift.
      if ((trial & 1) == 0) {
        baseline = benchmark.run(false);
        callback = benchmark.run(true);
      } else {
        callback = benchmark.run(true);
        baseline = benchmark.run(false);
      }
      baseline_ms.push_back(baseline.stream_ms);
      callback_ms.push_back(callback.stream_ms);
      baseline_submit_us.push_back(baseline.submit_us);
      callback_submit_us.push_back(callback.submit_us);
      device_overhead_us.push_back(
          (callback.stream_ms - baseline.stream_ms) * 1000.0 / callback_count);
      submit_overhead_us.push_back(
          (callback.submit_us - baseline.submit_us) / callback_count);
      api_ns.insert(api_ns.end(), callback.api_ns.begin(), callback.api_ns.end());
      queue_us.insert(queue_us.end(), callback.queue_us.begin(), callback.queue_us.end());
      callback_body_us.insert(callback_body_us.end(), callback.callback_us.begin(),
                              callback.callback_us.end());
    }

    const Stats baseline_stream = stats(baseline_ms);
    const Stats callback_stream = stats(callback_ms);
    const Stats device_overhead = stats(device_overhead_us);
    const Stats baseline_submit = stats(baseline_submit_us);
    const Stats callback_submit = stats(callback_submit_us);
    const Stats submit_overhead = stats(submit_overhead_us);
    const Stats api = stats(api_ns);
    const Stats queue = stats(queue_us);
    const Stats callback_body = stats(callback_body_us);

    if (options.csv) {
      std::cout
          << "device,driver_version,runtime_version,kernels,callbacks,kernel_ci,elements,trials,cpu_load_threads,"
             "gpu_load_streams,baseline_stream_ms_mean,callback_stream_ms_mean,"
             "device_overhead_us_mean,device_overhead_us_p50,device_overhead_us_p95,"
             "host_api_ns_mean,host_api_ns_p50,host_api_ns_p95,"
             "host_submit_overhead_us_mean,queue_latency_us_mean,callback_body_us_mean\n";
      std::cout << '"' << properties.name << "\"," << driver_version << ','
                << runtime_version << ',' << options.kernels << ','
                << callback_count << ",\"" << ci_description(options) << "\","
                << options.elements << ',' << options.trials << ','
                << options.cpu_load_threads << ',' << options.gpu_load_streams << ','
                << baseline_stream.mean << ',' << callback_stream.mean << ','
                << device_overhead.mean << ',' << device_overhead.median << ','
                << device_overhead.p95 << ',' << api.mean << ',' << api.median << ','
                << api.p95 << ',' << submit_overhead.mean << ',' << queue.mean << ','
                << callback_body.mean << '\n';
      return 0;
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "cudaLaunchHostFunc overhead benchmark\n"
              << "Device:                 " << properties.name << " (" << options.device << ")\n"
              << "CUDA driver/runtime:    " << driver_version << " / " << runtime_version << "\n"
              << "Workload:               " << options.kernels << " kernels, "
              << callback_count << " callbacks, " << options.elements << " elements/kernel\n"
              << "Kernel CI (FLOP/byte):  " << ci_description(options) << "\n"
              << "Load:                   " << options.cpu_load_threads << " CPU threads, "
              << options.gpu_load_streams << " GPU streams x " << options.gpu_load_kernels
              << " kernels\n"
              << "Samples:                " << options.trials << " paired trials, "
              << api_ns.size() << " API calls\n\n";

    std::cout << "Host-side overhead\n"
              << "  cudaLaunchHostFunc API (ns)     mean " << api.mean << ", p50 "
              << api.median << ", p95 " << api.p95 << ", sd " << api.stdev << '\n'
              << "  Incremental submit/callback (us) mean " << submit_overhead.mean
              << ", p50 " << submit_overhead.median << ", p95 " << submit_overhead.p95
              << '\n'
              << "  Baseline/test submit total (us) " << baseline_submit.mean << " / "
              << callback_submit.mean << '\n'
              << "  steady_clock pair floor (ns)    " << timer_floor << "\n\n";

    std::cout << "GPU-visible stream overhead\n"
              << "  Incremental stall/callback (us) mean " << device_overhead.mean
              << ", p50 " << device_overhead.median << ", p95 " << device_overhead.p95
              << ", sd " << device_overhead.stdev << '\n'
              << "  Baseline/test stream total (ms) " << baseline_stream.mean << " / "
              << callback_stream.mean << "\n\n";

    std::cout << "Callback observations (not pure overhead)\n"
              << "  API-return to callback-start (us) mean " << queue.mean << ", p50 "
              << queue.median << ", p95 " << queue.p95 << '\n'
              << "  Callback body duration (us)          mean " << callback_body.mean
              << ", p50 " << callback_body.median << ", p95 " << callback_body.p95
              << '\n';
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
