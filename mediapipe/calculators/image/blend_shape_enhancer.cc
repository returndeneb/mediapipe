#include "mediapipe/calculators/image/blend_shape_enhancer.h"

#include "absl/log/absl_log.h"
#include "mediapipe/framework/formats/classification.pb.h" // For the ClassificationList structure
#include "mediapipe/framework/formats/tensor.h"            // Include Tensor for handling tensor operations
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/ret_check.h"

namespace mediapipe {

::mediapipe::Status BlendShapeEnhancer::GetContract(CalculatorContract* cc) {
    // Expecting ClassificationList input for blend shapes
    cc->Inputs().Tag("BLENDSHAPES").Set<mediapipe::ClassificationList>();
    
    // Expecting a vector of Tensors
    cc->Inputs().Tag("TENSORS").Set<std::vector<mediapipe::Tensor>>();  
    
    // The output will also be a ClassificationList
    cc->Outputs().Tag("BLENDSHAPES").Set<mediapipe::ClassificationList>(); 
    return ::mediapipe::OkStatus();
}

::mediapipe::Status BlendShapeEnhancer::Process(CalculatorContext* cc) {
    // Check inputs
    RET_CHECK(!cc->Inputs().Tag("BLENDSHAPES").IsEmpty()) << "BlendShapes input is empty.";
    RET_CHECK(!cc->Inputs().Tag("TENSORS").IsEmpty()) << "Tensors input is empty."; // Ensure tensors input is not empty

    // Retrieve blendshapes and the vector of output tensors
    const auto& blendshapes = cc->Inputs().Tag("BLENDSHAPES").Get<mediapipe::ClassificationList>();
    const auto& output_tensors = cc->Inputs().Tag("TENSORS").Get<std::vector<mediapipe::Tensor>>(); // Access vector of tensors

    // If there are any tensors, let’s print each value
    for (size_t i = 0; i < output_tensors.size(); ++i) {
        const auto& tensor = output_tensors[i];  // Reference the specific tensor
        const float* tensor_data = tensor.GetCpuReadView().buffer<float>(); // Access the raw float data
        
        // Logging the values in the tensor
        LOG(INFO) << "Output Tensor[" << i << "] Values:";
        
        for (int j = 0; j < 1; ++j) { // Assuming tensor has proper dimensions
            std::cout << "Value[" << j << "]: " << tensor_data[j]; // Print each value
        }
    }

    // Create an updated copy of blend shapes
    mediapipe::ClassificationList updated_blendshapes = blendshapes;

    // Validate classification size before attempting to update
    if (updated_blendshapes.classification_size() > 23) { // Ensure at least 24 classifications exist
        auto* class_entry = updated_blendshapes.mutable_classification(23); // Get mutable reference to entry at index 23
        // Updating with the first value of the first tensor as an example
        class_entry->set_score(output_tensors[0].GetCpuReadView().buffer<float>()[0]); // Set the updated score 
    } else {
        LOG(WARNING) << "BlendShapes does not have an index 23. Current number of classifications: " 
                     << updated_blendshapes.classification_size(); // Log warning for too few classifications
    }

    // Emit the updated blend shapes
    cc->Outputs().Tag("BLENDSHAPES").Add(new mediapipe::ClassificationList(updated_blendshapes), cc->InputTimestamp());
    return ::mediapipe::OkStatus();
}

// Register the BlendShapeEnhancer calculator
REGISTER_CALCULATOR(BlendShapeEnhancer);

}  // namespace mediapipe
