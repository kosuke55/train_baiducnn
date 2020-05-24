/*
 * Copyright 2020 TierIV. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "feature_generator_pb.h"

FeatureGenerator::FeatureGenerator(int range, int width, int height)

    : min_height_(-5.0), max_height_(5.0), range_(range), width_(width), height_(height) {
  log_table_.resize(256);
  for (size_t i = 0; i < log_table_.size(); ++i) {
    log_table_[i] = std::log1p(static_cast<float>(i));
  }

  int siz = height_ * width_;
}

float FeatureGenerator::logCount(int count) {
  if (count < static_cast<int>(log_table_.size())) {
    return log_table_[count];
  }
  return std::log(static_cast<float>(1 + count));
}

std::vector<float> FeatureGenerator::generate(const pybind11::array_t<float> points, const bool use_constant_feature,
                                              const bool use_intensity_feature) {
  const auto& buff_info = points.request();
  const auto& shape = buff_info.shape;

  int siz = height_ * width_;
  std::vector<float> in_feature;

  float *max_height_data, *direction_data, *mean_height_data, *distance_data, *count_data, *top_intensity_data,
      *mean_intensity_data, *nonempty_data;

  int channels;
  if (use_constant_feature && use_intensity_feature) {
    channels = 8;
    in_feature.resize(siz * channels);
    max_height_data = &in_feature[0];
    direction_data = &in_feature[0] + siz * 3;
    top_intensity_data = &in_feature[0] + siz * 4;
    mean_intensity_data = &in_feature[0] + siz * 5;
    distance_data = &in_feature[0] + siz * 6;
    nonempty_data = &in_feature[0] + siz * 7;
  } else if (use_constant_feature) {
    channels = 6;
    in_feature.resize(siz * channels);
    max_height_data = &in_feature[0];
    direction_data = &in_feature[0] + siz * 3;
    distance_data = &in_feature[0] + siz * 4;
    nonempty_data = &in_feature[0] + siz * 5;
  } else if (use_intensity_feature) {
    channels = 6;
    in_feature.resize(siz * channels);
    max_height_data = &in_feature[0];
    top_intensity_data = &in_feature[0] + siz * 3;
    mean_intensity_data = &in_feature[0] + siz * 4;
    nonempty_data = &in_feature[0] + siz * 5;
  } else {
    channels = 4;
    in_feature.resize(siz * channels);
    max_height_data = &in_feature[0];
    nonempty_data = &in_feature[0] + siz * 3;
  }

  mean_height_data = &in_feature[0] + siz;
  count_data = &in_feature[0] + siz * 2;

  if (use_constant_feature) {
    for (int row = 0; row < height_; ++row) {
      for (int col = 0; col < width_; ++col) {
        int idx = row * width_ + col;
        float center_x = Pixel2Pc(row, height_, range_);
        float center_y = Pixel2Pc(col, width_, range_);
        constexpr double K_CV_PI = 3.1415926535897932384626433832795;
        direction_data[idx] = static_cast<float>(std::atan2(center_y, center_x) / (2.0 * K_CV_PI));
        distance_data[idx] = static_cast<float>(std::hypot(center_x, center_y) / 60.0 - 0.5);
      }
    }
  }

  for (int i = 0; i < siz * channels; ++i) {
    if (i < siz) {
      in_feature[i] = -0.5;
    } else {
      in_feature[i] = 0;
    }
  }

  map_idx_.resize(shape[0]);
  float inv_res_x = 0.5 * static_cast<float>(width_) / static_cast<float>(range_);
  float inv_res_y = 0.5 * static_cast<float>(height_) / static_cast<float>(range_);

  for (auto i = 0; i < shape[0]; ++i) {
    if (*points.data(i, 2) <= min_height_ || *points.data(i, 2) >= max_height_) {
      map_idx_[i] = -1;
      continue;
    }
    int pos_x = F2I(*points.data(i, 1), range_, inv_res_x);
    int pos_y = F2I(*points.data(i, 0), range_, inv_res_y);
    if (pos_x >= width_ || pos_x < 0 || pos_y >= height_ || pos_y < 0) {
      map_idx_[i] = -1;
      continue;
    }
    map_idx_[i] = pos_y * width_ + pos_x;
    int idx = map_idx_[i];
    float pz = *points.data(i, 2);
    float pi = *points.data(i, 3) / 255.0;

    if (max_height_data[idx] < pz) {
      max_height_data[idx] = pz;
      top_intensity_data[idx] = pi;
    }
    mean_height_data[idx] += static_cast<float>(pz);
    mean_intensity_data[idx] += static_cast<float>(pi);
    count_data[idx] += static_cast<float>(1);
  }
  for (int i = 0; i < siz; ++i) {
    constexpr double EPS = 1e-6;
    if (count_data[i] < EPS) {
      max_height_data[i] = static_cast<float>(0);
    } else {
      mean_height_data[i] /= count_data[i];
      mean_intensity_data[i] /= count_data[i];
      nonempty_data[i] = static_cast<float>(1);
    }
    count_data[i] = logCount(static_cast<int>(count_data[i]));
  }

  return in_feature;
}

namespace py = pybind11;
PYBIND11_PLUGIN(feature_generator_pb) {
  py::module m("feature_generator_pb", "feature_generator made by pybind11");

  py::class_<FeatureGenerator>(m, "FeatureGenerator")
      .def(py::init<int, int, int>())
      .def("generate", &FeatureGenerator::generate);

  return m.ptr();
}
