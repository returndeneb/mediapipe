#include "mediapipe/calculators/image/cheek_crop_calculator.h"

#include <cmath>
#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"  // For NormalizedLandmarkList
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status.h"

namespace mediapipe {

::mediapipe::Status CheekCropCalculator::GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("LANDMARKS").Set<mediapipe::NormalizedLandmarkList>();  // NormalizedLandmarkList type
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status CheekCropCalculator::Process(CalculatorContext* cc) {
    RET_CHECK(!cc->Inputs().Tag("LANDMARKS").IsEmpty())
        << "Landmark input stream is empty.";
    RET_CHECK(!cc->Inputs().Tag("IMAGE").IsEmpty())
        << "Image input stream is empty.";

    // Get landmarks
    const auto& landmark_list = cc->Inputs().Tag("LANDMARKS").Get<mediapipe::NormalizedLandmarkList>();
    const auto& input_frame = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
    cv::Mat image = formats::MatView(&input_frame);

    if (landmark_list.landmark_size() <= 0) {
        return ::mediapipe::UnknownError("No landmarks available.");  // Correct approach
    }

    int left_cheek_index = 192;  // Example indices; adjust according to your landmarks setup
    int right_cheek_index = 416;

    const auto& left_cheek_landmark = landmark_list.landmark(left_cheek_index);
    const auto& right_cheek_landmark = landmark_list.landmark(right_cheek_index);

    int center_x_left = static_cast<int>(left_cheek_landmark.x() * image.cols);
    int center_y_left = static_cast<int>(left_cheek_landmark.y() * image.rows);
    int center_x_right = static_cast<int>(right_cheek_landmark.x() * image.cols);
    int center_y_right = static_cast<int>(right_cheek_landmark.y() * image.rows);

    cv::Rect left_cheek_rect(center_x_left - 50, center_y_left - 50, 100, 100);
    cv::Rect right_cheek_rect(center_x_right - 50, center_y_right - 50, 100, 100);

    left_cheek_rect &= cv::Rect(0, 0, image.cols, image.rows);
    right_cheek_rect &= cv::Rect(0, 0, image.cols, image.rows);

    cv::Mat left_cheek = image(left_cheek_rect);
    cv::Mat right_cheek = image(right_cheek_rect);

    cv::Mat combined_cheeks(100, 200, image.type());
    left_cheek.copyTo(combined_cheeks(cv::Rect(0, 0, left_cheek.cols, left_cheek.rows)));
    right_cheek.copyTo(combined_cheeks(cv::Rect(100, 0, right_cheek.cols, right_cheek.rows)));

    auto output_frame = absl::make_unique<ImageFrame>(
        input_frame.Format(), combined_cheeks.cols, combined_cheeks.rows);
    cv::Mat output_mat = formats::MatView(output_frame.get());
    combined_cheeks.copyTo(output_mat);

    cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(CheekCropCalculator);

}  // namespace mediapipe
