#ifndef MEDIAPIPE_CALCULATORS_UTIL_FACE_BLENDSHAPES_PRINTER_H_
#define MEDIAPIPE_CALCULATORS_UTIL_FACE_BLENDSHAPES_PRINTER_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"

namespace mediapipe {

// FaceBlendshapesPrinter is a calculator that prints the data in a
// ClassificationList coming from the FACE_BLENDSHAPES stream.
class FaceBlendshapesPrinter : public CalculatorBase {
 public:
  FaceBlendshapesPrinter() = default;
  ~FaceBlendshapesPrinter() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Process(CalculatorContext* cc) override;
  
 private:
  void PrintBlendshapes(const ClassificationList& blendshapes) const;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_UTIL_FACE_BLENDSHAPES_PRINTER_H_
