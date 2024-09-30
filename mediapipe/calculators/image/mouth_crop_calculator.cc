#include "mediapipe/calculators/image/mouth_crop_calculator.h"

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

::mediapipe::Status MouthCropCalculator::GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("LANDMARKS").Set<mediapipe::NormalizedLandmarkList>();  // NormalizedLandmarkList type
    cc->Inputs().Tag("IMAGE").Set<ImageFrame>();
    cc->Outputs().Tag("IMAGE").Set<ImageFrame>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status MouthCropCalculator::Process(CalculatorContext* cc) {
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

    // Compute bounding box for the face based on selected facial landmarks
    int x_min = image.cols, x_max = 0, y_min = image.rows, y_max = 0;

    // Indices for landmarks around the mouth.
    std::vector<int> face_landmark_indices = {432, 214, 164, 200};

    for (int index : face_landmark_indices) {
        const auto& landmark = landmark_list.landmark(index);
        int x = static_cast<int>(landmark.x() * image.cols);
        int y = static_cast<int>(landmark.y() * image.rows);
        x_min = std::min(x_min, x);
        x_max = std::max(x_max, x);
        y_min = std::min(y_min, y);
        y_max = std::max(y_max, y);
    }

    // Calculate the center
    int center_x = (x_min + x_max) / 2;
    int center_y = (y_min + y_max) / 2;

    // Calculate the size of the square bounding box
    int max_length = std::max(x_max - x_min, y_max - y_min);
    int half_length = max_length / 2;

    // Ensure the cropping stays within image bounds
    int start_x = std::max(center_x - half_length, 0);
    int start_y = std::max(center_y - half_length, 0);
    int end_x = std::min(center_x + half_length, image.cols);
    int end_y = std::min(center_y + half_length, image.rows);

    // Crop the image
    cv::Mat cropped_mouth = image(cv::Rect(start_x, start_y, end_x - start_x, end_y - start_y));

    // Resize to 32x32
    cv::resize(cropped_mouth, cropped_mouth, cv::Size(32, 32));

    auto output_frame = absl::make_unique<ImageFrame>(
        ImageFormat::SRGB, 32, 32);
    cv::Mat output_mat = formats::MatView(output_frame.get());
    cropped_mouth.copyTo(output_mat);

    cc->Outputs().Tag("IMAGE").Add(output_frame.release(), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(MouthCropCalculator);

}  // namespace mediapipe
