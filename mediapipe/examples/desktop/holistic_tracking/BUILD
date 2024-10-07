package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "Avakit_video",
    data = [
        "//mediapipe/modules/face_landmark:face_landmark.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/holistic_landmark:hand_recrop.tflite",
        "//mediapipe/modules/pose_detection:pose_detection.tflite",
        "//mediapipe/modules/pose_landmark:pose_landmark_full.tflite",
    ],
    deps = [
        "//mediapipe/examples/desktop:demo_run_graph_main",
        "//mediapipe/graphs/holistic_tracking:holistic_tracking_cpu_graph_deps"
    ],
    linkopts = [
        "ole32.lib",
        "mfplat.lib",
        "mf.lib",
        "mfuuid.lib",  # for GUID definitions
    ],
)