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
    // Check for empty input streams.
    if (cc->Inputs().Tag("LANDMARKS").IsEmpty()) {
        ABSL_LOG(WARNING) << "Skipping frame: Landmark input stream is empty.";
        return ::mediapipe::OkStatus();
    }

    if (cc->Inputs().Tag("IMAGE").IsEmpty()) {
        ABSL_LOG(WARNING) << "Skipping frame: Image input stream is empty.";
        return ::mediapipe::OkStatus();
    }

    // Retrieve the landmarks and image
    const auto& landmark_list = cc->Inputs().Tag("LANDMARKS").Get<mediapipe::NormalizedLandmarkList>();
    const auto& input_frame = cc->Inputs().Tag("IMAGE").Get<ImageFrame>();
    cv::Mat image = formats::MatView(&input_frame);

    if (landmark_list.landmark_size() < 264) {
        ABSL_LOG(WARNING) << "Not enough landmarks available.";
        return ::mediapipe::OkStatus();
    }

    // Define landmark indices for mouth and eyes
    int upper_lip_index = 13;
    int lower_lip_index = 14;
    int left_eye_index = 33;
    int right_eye_index = 263;

    auto get_landmark_coords = [&](int index) {
        const auto& landmark = landmark_list.landmark(index);
        return std::make_pair(static_cast<int>(landmark.x() * image.cols),
                            static_cast<int>(landmark.y() * image.rows));
    };

    auto [upper_lip_x, upper_lip_y] = get_landmark_coords(upper_lip_index);
    auto [lower_lip_x, lower_lip_y] = get_landmark_coords(lower_lip_index);
    auto [left_eye_x, left_eye_y] = get_landmark_coords(left_eye_index);  // Rename from `_` to `left_eye_y`
    auto [right_eye_x, right_eye_y] = get_landmark_coords(right_eye_index);  // Rename from `_` to `right_eye_y`


    // Compute mouth center
    int mouth_center_x = (upper_lip_x + lower_lip_x) / 2;
    int mouth_center_y = (upper_lip_y + lower_lip_y) / 2;

    // Calculate face width and derive crop dimensions
    int face_width = right_eye_x - left_eye_x;
    float face_size_ratio = 1;  // Adjust this as needed
    int crop_width = static_cast<int>(face_width * face_size_ratio);
    int crop_height = static_cast<int>(face_width * (face_size_ratio / 2));

    // Calculate crop rectangle and apply constraints
    int crop_x_min = std::max(0, mouth_center_x - crop_width / 2);
    int crop_y_min = std::max(0, mouth_center_y - crop_height / 2);
    int crop_x_max = std::min(image.cols, mouth_center_x + crop_width / 2);
    int crop_y_max = std::min(image.rows, mouth_center_y + crop_height / 2);

    // Crop and resize the mouth region
    cv::Rect mouth_rect(crop_x_min, crop_y_min, crop_x_max - crop_x_min, crop_y_max - crop_y_min);
    cv::Mat mouth_cropped = image(mouth_rect);
    cv::resize(mouth_cropped, mouth_cropped, cv::Size(64, 64));
    
    // Prepare the output frame
    auto output_frame = absl::make_unique<ImageFrame>(
        input_frame.Format(), mouth_cropped.cols, mouth_cropped.rows);
    cv::Mat output_mat = formats::MatView(output_frame.get());
    mouth_cropped.copyTo(output_mat);


    // cv::imshow("Original Image", image);
    // cv::imshow("Combined Cheeks", mouth_cropped);

    cv::waitKey(1);

    // Output the cropped mouth image
    cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}


REGISTER_CALCULATOR(CheekCropCalculator);

}  // namespace mediapipe
