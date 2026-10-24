// -----------------------------------------------------------------------------
// HybridRT | C++20 Model Inference Framework
// File: tests/core/canvas_test.cpp
// Description: Tests drawing of full-image and cropped results.
// -----------------------------------------------------------------------------

#include <hrt/core/canvas.hpp>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace {

TEST(CanvasTest, DrawsFullImageAndCroppedBoxes) {
  cv::Mat canvas = cv::Mat::zeros(64, 64, CV_8UC3);
  hrt::postprocess::DetectResultMap result{{0, {{{0, 0, 64, 64}, 1.0F, 0, -1}}}};
  hrt::core::draw(canvas, result);
  EXPECT_GT(cv::countNonZero(canvas.reshape(1)), 0);
  canvas.setTo(0);
  result.at(0)[0].rect = {8, 8, 48, 48};
  hrt::core::draw(canvas, result);
  EXPECT_GT(cv::countNonZero(canvas.reshape(1)), 0);
}

TEST(CanvasTest, DrawsFullImageKeypointsAndBorder) {
  cv::Mat canvas = cv::Mat::zeros(64, 64, CV_8UC3);
  hrt::postprocess::KeypointResultMap result;
  auto& object = result[0].emplace_back();
  object.box.rect = {0, 0, 64, 64};
  object.keypoints.push_back({{32, 32}});
  hrt::core::draw(canvas, result);
  EXPECT_GT(cv::countNonZero(canvas.reshape(1)), 0);
  EXPECT_GT(cv::countNonZero(canvas.row(0).reshape(1)), 0);
  EXPECT_GT(cv::countNonZero(canvas.col(0).reshape(1)), 0);
}

TEST(CanvasTest, DrawsFullImageMasksAndBorder) {
  cv::Mat canvas = cv::Mat::zeros(64, 64, CV_8UC3);
  hrt::postprocess::SegmentResultMap result;
  auto& object = result[0].emplace_back();
  object.box.rect = {0, 0, 64, 64};
  object.mask.rect = {0, 0, 64, 64};
  object.mask.data.resize(64 * 64, 0);
  object.mask.data[32 * 64 + 32] = 1;
  hrt::core::draw(canvas, result);
  EXPECT_GT(cv::countNonZero(canvas.reshape(1)), 3);
  EXPECT_GT(cv::countNonZero(canvas.row(0).reshape(1)), 0);
}

TEST(CanvasTest, DrawsClassifyLabelTopLeft) {
  cv::Mat canvas = cv::Mat::zeros(64, 128, CV_8UC3);
  hrt::core::draw(canvas, hrt::postprocess::ClassifyResult{{}, {0.1F, 0.9F}});
  EXPECT_GT(cv::countNonZero(canvas(cv::Rect(0, 0, 128, 32)).clone().reshape(1)), 0);
  EXPECT_EQ(cv::countNonZero(canvas(cv::Rect(0, 40, 128, 24)).clone().reshape(1)), 0);
}

} // namespace
