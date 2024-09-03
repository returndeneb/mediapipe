#include "face_blendshapes_printer.h"
#include <iostream>

namespace mediapipe {

absl::Status FaceBlendshapesPrinter::GetContract(CalculatorContract* cc) {
  // Declaring the input stream for face blendshapes as ClassificationList.
  cc->Inputs().Tag("FACE_BLENDSHAPES").Set<ClassificationList>();
  return absl::OkStatus();
}

absl::Status FaceBlendshapesPrinter::Process(CalculatorContext* cc) {
  // Check if the input stream has data.
  if (!cc->Inputs().Tag("FACE_BLENDSHAPES").IsEmpty()) {
    const auto& blendshapes = cc->Inputs()
                                 .Tag("FACE_BLENDSHAPES")
                                 .Get<ClassificationList>();
    PrintBlendshapes(blendshapes);
  }
  return absl::OkStatus();
}

void FaceBlendshapesPrinter::PrintBlendshapes(
    const ClassificationList& blendshapes) const {
//   std::cout << "Face Blendshapes:" << std::endl;
//   for (const auto& classification : blendshapes.classification()) {
//     std::cout << "  - Label: " << classification.label()
//               << ", Score: " << classification.score() << std::endl;
//   }
}

// Register the calculator with MediaPipe.
REGISTER_CALCULATOR(FaceBlendshapesPrinter);

}  // namespace mediapipe
