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

#ifndef FEATURE_GENERATOR_H
#define FEATURE_GENERATOR_H

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "util.h"

class FeatureGenerator {
 private:
  int width_ = 672;
  int height_ = 672;
  int range_ = 70;

  float min_height_ = 0.0;
  float max_height_ = 0.0;

  std::vector<float> log_table_;
  std::vector<float> in_feature_;

  std::vector<int> map_idx_;

  float logCount(int count);

 public:
  FeatureGenerator(int range, int width, int height);
  ~FeatureGenerator() {}
  std::vector<float> generate(const pybind11::array_t<float> points, const bool use_constant_feature,
                              const bool use_intensity_feature);
};

#endif  // FEATURE_GENERATOR_H
