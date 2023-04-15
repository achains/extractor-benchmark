#include "config.h"
#include "deplex/config.h"
#include "deplex/plane_extractor.h"
#include "deplex/utils/utils.h"

#include <filesystem>
#include <iostream>

void process(){
  // Sort data entries
  std::vector<std::filesystem::directory_entry> sorted_input_data;
  for (auto const& entry : std::filesystem::directory_iterator(benchmark_config::image_dir)) {
    sorted_input_data.push_back(entry);
  }
  sort(sorted_input_data.begin(), sorted_input_data.end());

  // Benchmark
  std::vector<size_t> time_vector;
  auto algorithm = deplex::PlaneExtractor(480, 640, deplex::config::Config(benchmark_config::deplex_config));
  for (auto const& entry: sorted_input_data){
    auto START_TIME = std::chrono::high_resolution_clock::now();
    auto img = deplex::utils::Image(entry.path());
    algorithm.process(img.toPointCloud(deplex::utils::readIntrinsics(benchmark_config::deplex_intrinsics)));
    auto FINISH_TIME = std::chrono::high_resolution_clock::now();
    time_vector.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(FINISH_TIME - START_TIME).count());
    std::cout << "Processed image: " << entry.path().filename() << " with time: " << time_vector.back() << '\n';
  }
}

int main(){
  process();
  return 0;
}