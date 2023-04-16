/*
 * Copyright 2018 Pedro Proenza <p.proenca@surrey.ac.uk> (University of Surrey)
 *
 */

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <numeric>
#define _USE_MATH_DEFINES
#include <dirent.h>
#include <math.h>
#include <Eigen/Dense>
#include <boost/algorithm/string.hpp>
#include <iterator>
#include <opencv2/opencv.hpp>
#include <string>
#include "CAPE.h"

#include "config.h"

using namespace std;

bool done = false;
bool cylinder_detection = false;
CAPE* plane_detector;
std::vector<cv::Vec3b> color_code;

bool loadCalibParameters(string filepath, cv::Mat& intrinsics_rgb, cv::Mat& dist_coeffs_rgb, cv::Mat& intrinsics_ir,
                         cv::Mat& dist_coeffs_ir, cv::Mat& R, cv::Mat& T) {
  cv::FileStorage fs(filepath, cv::FileStorage::READ);
  if (fs.isOpened()) {
    fs["RGB_intrinsic_params"] >> intrinsics_rgb;
    fs["RGB_distortion_coefficients"] >> dist_coeffs_rgb;
    fs["IR_intrinsic_params"] >> intrinsics_ir;
    fs["IR_distortion_coefficients"] >> dist_coeffs_ir;
    fs["Rotation"] >> R;
    fs["Translation"] >> T;
    fs.release();
    return true;
  } else {
    cerr << "Calibration file missing" << endl;
    return false;
  }
}

void projectPointCloud(cv::Mat& X, cv::Mat& Y, cv::Mat& Z, cv::Mat& U, cv::Mat& V, float fx_rgb, float fy_rgb,
                       float cx_rgb, float cy_rgb, double z_min, Eigen::MatrixXf& cloud_array) {
  int width = X.cols;
  int height = X.rows;

  // Project to image coordinates
  cv::divide(X, Z, U, 1);
  cv::divide(Y, Z, V, 1);
  U = U * fx_rgb + cx_rgb;
  V = V * fy_rgb + cy_rgb;
  // Reusing U as cloud index
  // U = V*width + U + 0.5;

  float *sz, *sx, *sy, *u_ptr, *v_ptr, *id_ptr;
  float z, u, v;
  int id;
  for (int r = 0; r < height; r++) {
    sx = X.ptr<float>(r);
    sy = Y.ptr<float>(r);
    sz = Z.ptr<float>(r);
    u_ptr = U.ptr<float>(r);
    v_ptr = V.ptr<float>(r);
    for (int c = 0; c < width; c++) {
      z = sz[c];
      u = u_ptr[c];
      v = v_ptr[c];
      if (z > z_min && u > 0 && v > 0 && u < width && v < height) {
        id = floor(v) * width + u;
        cloud_array(id, 0) = sx[c];
        cloud_array(id, 1) = sy[c];
        cloud_array(id, 2) = z;
      }
    }
  }
}

void organizePointCloudByCell(Eigen::MatrixXf& cloud_in, Eigen::MatrixXf& cloud_out, cv::Mat& cell_map) {
  int width = cell_map.cols;
  int height = cell_map.rows;
  int mxn = width * height;
  int mxn2 = 2 * mxn;

  int id, it(0);
  int* cell_map_ptr;
  for (int r = 0; r < height; r++) {
    cell_map_ptr = cell_map.ptr<int>(r);
    for (int c = 0; c < width; c++) {
      id = cell_map_ptr[c];
      *(cloud_out.data() + id) = *(cloud_in.data() + it);
      *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
      *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
      it++;
    }
  }
}

int main(int argc, char** argv) {
  int max_frames = -1;
  if (argc > 1) {
    max_frames = std::stoi(argv[1]);
  }
  stringstream input_path(benchmark_config::image_dir);
  stringstream params_path(benchmark_config::cape_config);
  stringstream calib_path(benchmark_config::cape_intrinsics);

  if (argc > 2) {
    input_path << argv[1];
    params_path << argv[2];
  }
  // Get parameters
  if (params_path.str().empty()) {
    cout << "No parameters file specified. Using defaults." << endl;
  } else {
    readIni(params_path);
  }

  // Get intrinsics
  cv::Mat K_rgb, K_ir, dist_coeffs_rgb, dist_coeffs_ir, R_stereo, t_stereo;
  loadCalibParameters(calib_path.str(), K_rgb, dist_coeffs_rgb, K_ir, dist_coeffs_ir, R_stereo, t_stereo);
  float fx_ir = K_ir.at<double>(0, 0);
  float fy_ir = K_ir.at<double>(1, 1);
  float cx_ir = K_ir.at<double>(0, 2);
  float cy_ir = K_ir.at<double>(1, 2);
  float fx_rgb = K_rgb.at<double>(0, 0);
  float fy_rgb = K_rgb.at<double>(1, 1);
  float cx_rgb = K_rgb.at<double>(0, 2);
  float cy_rgb = K_rgb.at<double>(1, 2);

  // ============= INPUT DATA SORTING =============
  std::vector<std::filesystem::directory_entry> sorted_input_data;
  for (auto const& entry : std::filesystem::directory_iterator(input_path.str())) {
    sorted_input_data.push_back(entry);
  }
  sort(sorted_input_data.begin(), sorted_input_data.end());
  // ============= BENCHMARK =============
  std::vector<size_t> time_vector;
  for (auto const& entry : sorted_input_data) {
    // Read frame 1 to allocate and get dimension
    cv::Mat d_img;
    int width, height;
    stringstream image_path;
    stringstream depth_img_path;
    stringstream image_save_path;
    stringstream labels_save_path;

    d_img = cv::imread(entry.path().string(), cv::IMREAD_ANYDEPTH);
    if (d_img.data) {
      width = d_img.cols;
      height = d_img.rows;
    } else {
      cout << "Error loading file";
      return -1;
    }
    int nr_horizontal_cells = width / PATCH_SIZE;
    int nr_vertical_cells = height / PATCH_SIZE;

    // Pre-computations for backprojection
    cv::Mat_<float> X_pre(height, width);
    cv::Mat_<float> Y_pre(height, width);
    cv::Mat_<float> U(height, width);
    cv::Mat_<float> V(height, width);
    for (int r = 0; r < height; r++) {
      for (int c = 0; c < width; c++) {
        // Not efficient but at this stage doesn t matter
        X_pre.at<float>(r, c) = (c - cx_ir) / fx_ir;
        Y_pre.at<float>(r, c) = (r - cy_ir) / fy_ir;
      }
    }

    // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are
    // contiguous
    cv::Mat_<int> cell_map(height, width);

    for (int r = 0; r < height; r++) {
      int cell_r = r / PATCH_SIZE;
      int local_r = r % PATCH_SIZE;
      for (int c = 0; c < width; c++) {
        int cell_c = c / PATCH_SIZE;
        int local_c = c % PATCH_SIZE;
        cell_map.at<int>(r, c) =
            (cell_r * nr_horizontal_cells + cell_c) * PATCH_SIZE * PATCH_SIZE + local_r * PATCH_SIZE + local_c;
      }
    }

    cv::Mat_<float> X(height, width);
    cv::Mat_<float> Y(height, width);
    Eigen::MatrixXf cloud_array(width * height, 3);
    Eigen::MatrixXf cloud_array_organized(width * height, 3);
    // Populate with random color codes
    for (int i = 0; i < 100; i++) {
      cv::Vec3b color;
      color[0] = rand() % 255;
      color[1] = rand() % 255;
      color[2] = rand() % 255;
      color_code.push_back(color);
    }
    // Add specific colors for planes
    color_code[0][0] = 0;
    color_code[0][1] = 0;
    color_code[0][2] = 255;
    color_code[1][0] = 255;
    color_code[1][1] = 0;
    color_code[1][2] = 204;
    color_code[2][0] = 255;
    color_code[2][1] = 100;
    color_code[2][2] = 0;
    color_code[3][0] = 0;
    color_code[3][1] = 153;
    color_code[3][2] = 255;
    // Add specific colors for cylinders
    color_code[50][0] = 178;
    color_code[50][1] = 255;
    color_code[50][2] = 0;
    color_code[51][0] = 255;
    color_code[51][1] = 0;
    color_code[51][2] = 51;
    color_code[52][0] = 0;
    color_code[52][1] = 255;
    color_code[52][2] = 51;
    color_code[53][0] = 153;
    color_code[53][1] = 0;
    color_code[53][2] = 255;

    int frame_num = 0;
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(input_path.str().c_str())) != NULL) {
      while ((ent = readdir(dir)) != NULL) {
        if (boost::algorithm::contains(ent->d_name, ".png")) frame_num++;
      }
      closedir(dir);
    } else {
      perror("could not open directory");
      return EXIT_FAILURE;
    }

    // Initialize CAPE
    plane_detector = new CAPE(height, width, PATCH_SIZE, PATCH_SIZE, cylinder_detection, COS_ANGLE_MAX, MAX_MERGE_DIST);

    auto START_TIME = std::chrono::high_resolution_clock::now();

    d_img = cv::imread(entry.path().string(), cv::IMREAD_ANYDEPTH);
    d_img.convertTo(d_img, CV_32F);

    // Backproject to point cloud
    X = X_pre.mul(d_img);
    Y = Y_pre.mul(d_img);
    cloud_array.setZero();

    projectPointCloud(X, Y, d_img, U, V, fx_rgb, fy_rgb, cx_rgb, cy_rgb, t_stereo.at<double>(2), cloud_array);

    cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(height, width, cv::Vec3b(0, 0, 0));
    cv::Mat_<uchar> seg_output = cv::Mat_<uchar>(height, width, uchar(0));

    // Run CAPE
    int nr_planes, nr_cylinders;
    vector<PlaneSeg> plane_params;
    vector<CylinderSeg> cylinder_params;
    organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);
    plane_detector->process(cloud_array_organized, nr_planes, nr_cylinders, seg_output, plane_params, cylinder_params);
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
  std::ofstream output(std::string(benchmark_config::output_dir) + "/cape_benchmark.txt");
  std::copy(time_vector.begin(), time_vector.end(), std::ostream_iterator<size_t>(output, ","));

  return 0;
}
