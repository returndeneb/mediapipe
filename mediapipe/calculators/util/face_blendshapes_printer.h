#ifndef MEDIAPIPE_CALCULATORS_UTIL_FACE_BLENDSHAPES_PRINTER_H_
#define MEDIAPIPE_CALCULATORS_UTIL_FACE_BLENDSHAPES_PRINTER_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h" // Include for NormalizedRect
#include <mediapipe/util/json.hpp>  // Include for JSON handling

namespace mediapipe {

// FaceBlendshapesPrinter is a calculator that processes multiple input streams
// and outputs JSON-encoded data of various landmarks and image size.
class FaceBlendshapesPrinter : public CalculatorBase {
 public:
  FaceBlendshapesPrinter() = default;
  ~FaceBlendshapesPrinter() override = default;

  // Sets up the inputs and outputs for the calculator.
  static absl::Status GetContract(CalculatorContract* cc);

  // Processes the incoming data streams.
  absl::Status Process(CalculatorContext* cc) override;

 private:
  // Converts a ClassificationList to a JSON array (if used, adjust as needed).
  nlohmann::json ClassificationListToJson(const ClassificationList& classification_list) const;

  // Converts a NormalizedLandmarkList to a JSON array.
  nlohmann::json NormalizedLandmarkListToJson(const NormalizedLandmarkList& landmarks) const;
  nlohmann::json BodyLandmarksToJson(const NormalizedLandmarkList& landmarks, float ratio) const;
  nlohmann::json FaceLandmarksToJson(const NormalizedLandmarkList& landmarks, float ratio, const std::vector<int>& face_landmarks_indices) const;
  nlohmann::json HandLandmarksToJson(const NormalizedLandmarkList& landmarks, float ratio) const;

  static const std::vector<int> FACE_LANDMARKS;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_FACE_BLENDSHAPES_PRINTER_H_