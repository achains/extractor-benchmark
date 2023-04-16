//
// Copyright 2014 Mitsubishi Electric Research Laboratories All
// Rights Reserved.
//
// Permission to use, copy and modify this software and its
// documentation without fee for educational, research and non-profit
// purposes, is hereby granted, provided that the above copyright
// notice, this paragraph, and the following three paragraphs appear
// in all copies.
//
// To request permission to incorporate this software into commercial
// products contact: Director; Mitsubishi Electric Research
// Laboratories (MERL); 201 Broadway; Cambridge, MA 02139.
//
// IN NO EVENT SHALL MERL BE LIABLE TO ANY PARTY FOR DIRECT,
// INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING
// LOST PROFITS, ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS
// DOCUMENTATION, EVEN IF MERL HAS BEEN ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGES.
//
// MERL SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE. THE SOFTWARE PROVIDED HEREUNDER IS ON AN
// "AS IS" BASIS, AND MERL HAS NO OBLIGATIONS TO PROVIDE MAINTENANCE,
// SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
//
#pragma warning(disable : 4996)
#pragma warning(disable : 4819)
#define _CRT_SECURE_NO_WARNINGS

#include <deplex/utils/utils.h>
#include "config.h"

#include <filesystem>
#include <iterator>
#include <map>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <pcl/common/transforms.h>

#include <opencv2/core/eigen.hpp>
#include "opencv2/opencv.hpp"

#include "AHCPlaneFitter.hpp"
using ahc::utils::Timer;

// pcl::PointCloud interface for our ahc::PlaneFitter
template <class PointT>
struct OrganizedImage3D {
  const pcl::PointCloud<PointT>& cloud;
  // note: ahc::PlaneFitter assumes mm as unit!!!
  const double unitScaleFactor;

  OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1) {}

  OrganizedImage3D(const OrganizedImage3D& other) : cloud(other.cloud), unitScaleFactor(other.unitScaleFactor) {}

  inline int width() const { return cloud.width; }

  inline int height() const { return cloud.height; }

  inline bool get(const int row, const int col, double& x, double& y, double& z) const {
    const PointT& pt = cloud.at(col, row);
    x = pt.x * unitScaleFactor;
    y = pt.y * unitScaleFactor;
    z = pt.z * unitScaleFactor;  // TODO: will this slowdown the speed?
    return z == 0;               // return false if current depth is NaN
  }
};

typedef OrganizedImage3D<pcl::PointXYZ> ImageXYZ;
typedef ahc::PlaneFitter<ImageXYZ> PlaneFitter;
typedef pcl::PointCloud<pcl::PointXYZRGB> CloudXYZRGB;

namespace global {
std::map<std::string, std::string> ini;
PlaneFitter pf;
bool showWindow = true;

#ifdef _WIN32
const char filesep = '\\';
#else
const char filesep = '/';
#endif

// similar to matlab's fileparts
// if in=parent/child/file.txt
// then path=parent/child
// name=file, ext=txt
void fileparts(const std::string& str, std::string* pPath = 0, std::string* pName = 0, std::string* pExt = 0) {
  std::string::size_type last_sep = str.find_last_of(filesep);
  std::string::size_type last_dot = str.find_last_of('.');
  if (last_dot < last_sep)  // "D:\parent\child.folderA\file", "D:\parent\child.folderA\"
    last_dot = std::string::npos;

  std::string path, name, ext;

  if (last_sep == std::string::npos) {
    path = ".";
    if (last_dot == std::string::npos) {  // "test"
      name = str;
      ext = "";
    } else {  // "test.txt"
      name = str.substr(0, last_dot);
      ext = str.substr(last_dot + 1);
    }
  } else {
    path = str.substr(0, last_sep);
    if (last_dot == std::string::npos) {  // "d:/parent/test", "d:/parent/child/"
      name = str.substr(last_sep + 1);
      ext = "";
    } else {  // "d:/parent/test.txt"
      name = str.substr(last_sep + 1, last_dot - last_sep - 1);
      ext = str.substr(last_dot + 1);
    }
  }

  if (pPath != 0) {
    *pPath = path;
  }
  if (pName != 0) {
    *pName = name;
  }
  if (pExt != 0) {
    *pExt = ext;
  }
}

//"D:/test/test.txt" -> "D:/test/"
std::string getFileDir(const std::string& fileName) {
  std::string path;
  fileparts(fileName, &path);
  return path;
}

//"D:/parent/test.txt" -> "test"
//"D:/parent/test" -> "test"
std::string getNameNoExtension(const std::string& fileName) {
  std::string name;
  fileparts(fileName, 0, &name);
  return name;
}

void iniLoad(std::string iniFileName) {
  std::ifstream in(iniFileName);
  if (!in.is_open()) {
    std::cout << "[iniLoad] " << iniFileName << " not found, use default parameters!" << std::endl;
    return;
  }
  while (in) {
    std::string line;
    std::getline(in, line);
    if (line.empty() || line[0] == '#') continue;
    std::string key, value;
    size_t eqPos = line.find_first_of("=");
    if (eqPos == std::string::npos || eqPos == 0) {
      std::cout << "[iniLoad] ignore line:" << line << std::endl;
      continue;
    }
    key = line.substr(0, eqPos);
    value = line.substr(eqPos + 1);
    std::cout << "[iniLoad] " << key << "=>" << value << std::endl;
    ini[key] = value;
  }
}

template <class T>
T iniGet(std::string key, T default_value) {
  std::map<std::string, std::string>::const_iterator itr = ini.find(key);
  if (itr != ini.end()) {
    std::stringstream ss;
    ss << itr->second;
    T ret;
    ss >> ret;
    return ret;
  }
  return default_value;
}

template <>
std::string iniGet(std::string key, std::string default_value) {
  std::map<std::string, std::string>::const_iterator itr = ini.find(key);
  if (itr != ini.end()) {
    return itr->second;
  }
  return default_value;
}
}  // namespace global

void processOneFrame(pcl::PointCloud<pcl::PointXYZ>& cloud, const std::string& outputFilePrefix) {
  using global::pf;
  cv::Mat seg(cloud.height, cloud.width, CV_8UC3);

  // run PlaneFitter on the current frame of point cloud
  ImageXYZ Ixyz(cloud);
  pf.run(&Ixyz, 0, &seg, 0, false);
}

int process(int max_frames) {
  const auto unitScaleFactor = global::iniGet<double>("unitScaleFactor", 1.0f);

  using global::pf;
  // setup fitter
  pf.minSupport = global::iniGet<int>("minSupport", 3000);
  pf.windowWidth = global::iniGet<int>("windowWidth", 10);
  pf.windowHeight = global::iniGet<int>("windowHeight", 10);
  pf.doRefine = global::iniGet<int>("doRefine", 1) != 0;

  pf.params.initType = (ahc::InitType)global::iniGet("initType", (int)pf.params.initType);

  // T_mse
  pf.params.stdTol_merge = global::iniGet("stdTol_merge", pf.params.stdTol_merge);
  pf.params.stdTol_init = global::iniGet("stdTol_init", pf.params.stdTol_init);
  pf.params.depthSigma = global::iniGet("depthSigma", pf.params.depthSigma);

  // T_dz
  pf.params.depthAlpha = global::iniGet("depthAlpha", pf.params.depthAlpha);
  pf.params.depthChangeTol = global::iniGet("depthChangeTol", pf.params.depthChangeTol);

  // T_ang
  pf.params.z_near = global::iniGet("z_near", pf.params.z_near);
  pf.params.z_far = global::iniGet("z_far", pf.params.z_far);
  pf.params.angle_near = MACRO_DEG2RAD(global::iniGet("angleDegree_near", MACRO_RAD2DEG(pf.params.angle_near)));
  pf.params.angle_far = MACRO_DEG2RAD(global::iniGet("angleDegree_far", MACRO_RAD2DEG(pf.params.angle_far)));
  pf.params.similarityTh_merge =
      std::cos(MACRO_DEG2RAD(global::iniGet("similarityDegreeTh_merge", MACRO_RAD2DEG(pf.params.similarityTh_merge))));
  pf.params.similarityTh_refine = std::cos(
      MACRO_DEG2RAD(global::iniGet("similarityDegreeTh_refine", MACRO_RAD2DEG(pf.params.similarityTh_refine))));

#if defined(DEBUG_INIT) || defined(DEBUG_CLUSTER)
  pf.saveDir = outputDir;
  {  // create debug result folder
#ifdef _WIN32
    std::string cmd = "mkdir " + pf.saveDir + "\\output 2> NUL";
#else
    std::string cmd = "mkdir -p " + pf.saveDir + "\\output";
#endif
    system(cmd.c_str());
    std::cout << "create:" << (pf.saveDir + "\\output") << std::endl;
  }
#endif

  // Sort data entries
  std::vector<std::filesystem::directory_entry> sorted_input_data;
  for (auto const& entry : std::filesystem::directory_iterator(benchmark_config::image_dir)) {
    sorted_input_data.push_back(entry);
  }
  sort(sorted_input_data.begin(), sorted_input_data.end());

  // Benchmark
  std::vector<size_t> time_vector;
  for (auto const& entry : sorted_input_data) {
    auto START_TIME = std::chrono::high_resolution_clock::now();
    auto img = deplex::utils::Image(entry.path());
    pcl::PointCloud<pcl::PointXYZ> cloud;
    Eigen::MatrixXf pcd_points = img.toPointCloud(deplex::utils::readIntrinsics(benchmark_config::peac_intrinsics));
    for (const auto& point : pcd_points.rowwise()) {
      cloud.points.emplace_back(point[0], point[1], point[2]);
    }
    processOneFrame(cloud, "");
    auto FINISH_TIME = std::chrono::high_resolution_clock::now();
    time_vector.emplace_back(std::chrono::duration_cast<std::chrono::microseconds>(FINISH_TIME - START_TIME).count());
#ifdef BENCHMARK_VERBOSE
    std::cout << "Processed image: " << entry.path().filename() << " with time: " << time_vector.back() << '\n';
#endif
    if (time_vector.size() == max_frames) {
      break;
    }
  }

  // Output
  std::ofstream output(std::string(benchmark_config::output_dir) + "/peac_benchmark.txt");
  std::copy(time_vector.begin(), time_vector.end(), std::ostream_iterator<size_t>(output, ","));

  return 0;
}

int main(const int argc, const char** argv) {
  int max_frames = -1;
  if (argc > 1) {
    max_frames = std::stoi(argv[1]);
  }
  global::iniLoad(benchmark_config::peac_config);
  return process(max_frames);
}