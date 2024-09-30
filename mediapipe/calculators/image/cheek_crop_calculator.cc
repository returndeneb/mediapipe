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
    // Check if landmarks or image are empty, and skip processing if either is empty.
    if (cc->Inputs().Tag("LANDMARKS").IsEmpty()) {
        ABSL_LOG(WARNING) << "Skipping frame: Landmark input stream is empty.";
        return ::mediapipe::OkStatus();  // Skip this frame and return success.
    }
    
    if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
        ABSL_LOG(WARNING) << "Skipping frame: Image input stream is empty.";
        return ::mediapipe::OkStatus();  // Skip this frame and return success.
    }
    
    const auto& landmark_list = cc->Inputs().Tag("LANDMARKS").Get<mediapipe::NormalizedLandmarkList>();
    const auto& input_frame = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
    cv::Mat image = formats::MatView(&input_frame);

    if (landmark_list.landmark_size() <= 0) {
        ABSL_LOG(WARNING) << "No landmarks available in this frame.";
        return ::mediapipe::OkStatus();  // Skip this frame and return success.
    }

    int left_cheek_index = 192;
    int right_cheek_index = 416;

    // Compute bounding box for face based on selected facial landmarks
    int x_min = image.cols, x_max = 0, y_min = image.rows, y_max = 0;
    std::vector<int> face_landmark_indices = {10, 338, 297, 332, 263, 61, 146, 91, 181, 84, 17};

    for (int index : face_landmark_indices) {
        const auto& landmark = landmark_list.landmark(index);
        int x = static_cast<int>(landmark.x() * image.cols);
        int y = static_cast<int>(landmark.y() * image.rows);
        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
    }

    int face_width = x_max - x_min;
    int face_height = y_max - y_min;
    int short_side = std::max(face_width, face_height) / 3;
    int long_side = short_side * 2;

    // Calculate the cropping coordinates for cheeks
    const auto& left_cheek_landmark = landmark_list.landmark(left_cheek_index);
    const auto& right_cheek_landmark = landmark_list.landmark(right_cheek_index);

    int center_x_left = static_cast<int>(left_cheek_landmark.x() * image.cols);
    int center_y_left = static_cast<int>(left_cheek_landmark.y() * image.rows);
    int center_x_right = static_cast<int>(right_cheek_landmark.x() * image.cols);
    int center_y_right = static_cast<int>(right_cheek_landmark.y() * image.rows);

    cv::Rect left_cheek_rect(center_x_left - short_side / 2, center_y_left - long_side / 2, short_side, long_side);
    cv::Rect right_cheek_rect(center_x_right - short_side / 2, center_y_right - long_side / 2, short_side, long_side);

    left_cheek_rect &= cv::Rect(0, 0, image.cols, image.rows);
    right_cheek_rect &= cv::Rect(0, 0, image.cols, image.rows);

    cv::Mat left_cheek = image(left_cheek_rect);
    cv::Mat right_cheek = image(right_cheek_rect);

    cv::resize(left_cheek, left_cheek, cv::Size(short_side, long_side));
    cv::resize(right_cheek, right_cheek, cv::Size(short_side, long_side));

    cv::Mat combined_cheeks(long_side, long_side, image.type(), cv::Scalar(0));
    left_cheek.copyTo(combined_cheeks(cv::Rect(0, 0, short_side, long_side)));
    right_cheek.copyTo(combined_cheeks(cv::Rect(short_side, 0, short_side, long_side)));

    cv::resize(combined_cheeks, combined_cheeks, cv::Size(64, 64));

    // cv::imshow("Original Image", image);
    // cv::Mat combined_cheeks_rgb;
    // cv::cvtColor(combined_cheeks, combined_cheeks_rgb, cv::COLOR_BGR2RGB);
    // cv::imshow("Combined Cheeks", combined_cheeks);

    // cv::Mat normalized_combined_cheeks;
    // combined_cheeks.convertTo(normalized_combined_cheeks, CV_32F, 1.0 / 255.0);
    // normalized_combined_cheeks = (normalized_combined_cheeks - 0.5f) * 2.0f;

    // cv::waitKey(1);

    auto output_frame = absl::make_unique<ImageFrame>(
        input_frame.Format(), combined_cheeks.cols, combined_cheeks.rows);
    cv::Mat output_mat = formats::MatView(output_frame.get());
    combined_cheeks.copyTo(output_mat);

    cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}


REGISTER_CALCULATOR(CheekCropCalculator);

}  // namespace mediapipe
