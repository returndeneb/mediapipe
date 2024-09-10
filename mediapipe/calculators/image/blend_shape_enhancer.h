#ifndef MEDIAPIPE_CALCULATORS_IMAGE_BLEND_SHAPE_ENHANCER_H_
#define MEDIAPIPE_CALCULATORS_IMAGE_BLEND_SHAPE_ENHANCER_H_

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/classification.pb.h"  // Include classification format

namespace mediapipe {

class BlendShapeEnhancer : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc);
  ::mediapipe::Status Process(CalculatorContext* cc) override;
};

}  // namespace mediapipe

#endif  // MEDIAPIPE_CALCULATORS_IMAGE_BLEND_SHAPE_ENHANCER_H_
