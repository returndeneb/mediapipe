#include <iostream>  // Include the iostream for std::cout
#include "mediapipe/calculators/image/blend_shape_enhancer.h"
#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/classification.pb.h"
#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

::mediapipe::Status BlendShapeEnhancer::GetContract(CalculatorContract* cc) {
    cc->Inputs().Tag("BLENDSHAPES").Set<mediapipe::ClassificationList>();
    cc->Inputs().Tag("TENSORS").Set<std::vector<mediapipe::Tensor>>();
    cc->Outputs().Tag("BLENDSHAPES").Set<mediapipe::ClassificationList>();
    return ::mediapipe::OkStatus();
}

::mediapipe::Status BlendShapeEnhancer::Process(CalculatorContext* cc) {
    RET_CHECK(!cc->Inputs().Tag("BLENDSHAPES").IsEmpty()) << "BlendShapes input is empty.";
    RET_CHECK(!cc->Inputs().Tag("TENSORS").IsEmpty()) << "Tensors input is empty.";

    // Create a mutable copy of blendshapes
    auto blendshapes = cc->Inputs().Tag("BLENDSHAPES").Get<mediapipe::ClassificationList>();

    const auto& output_tensors = cc->Inputs().Tag("TENSORS").Get<std::vector<mediapipe::Tensor>>();

    const auto& tensor = output_tensors[0];
    const float* tensor_data = tensor.GetCpuReadView().buffer<float>();

    // Assume blendshapes contains at least 6 entries and tensor_data is not empty
    if (blendshapes.classification_size() > 6 && tensor_data != nullptr) {
        // Update the 5th index with a value from tensor_data
        float new_score = tensor_data[0];
        blendshapes.mutable_classification(6)->set_score(new_score);
        blendshapes.mutable_classification(7)->set_score(new_score);
        blendshapes.mutable_classification(8)->set_score(new_score);

        // Output the new score to stdout
        std::cout << "Updated blendshape score at index 5: " << new_score << std::endl;
    }

    // Ensure the modified blendshapes are sent to the output
    cc->Outputs().Tag("BLENDSHAPES").Add(new mediapipe::ClassificationList(blendshapes), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}

REGISTER_CALCULATOR(BlendShapeEnhancer);

}  // namespace mediapipe
