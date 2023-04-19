#include "deplex/config.h"
#include "deplex/plane_extractor.h"
#include "deplex/utils/utils.h"

#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char* argv[]) {
  std::stringstream image_path;
  std::stringstream config_path;
  std::stringstream intrinsics_path;

  image_path << argv[1];
  config_path << argv[2];
  intrinsics_path << argv[3];

  auto algorithm = deplex::PlaneExtractor(480, 640, deplex::config::Config(config_path.str()));
  auto img = deplex::utils::DepthImage(image_path.str());
  auto labels = algorithm.process(img.toPointCloud(deplex::utils::readIntrinsics(intrinsics_path.str())));

  std::ofstream output_file("output/labels.csv");

  output_file << labels.format(deplex::utils::CSVFormat);

  return 0;
}