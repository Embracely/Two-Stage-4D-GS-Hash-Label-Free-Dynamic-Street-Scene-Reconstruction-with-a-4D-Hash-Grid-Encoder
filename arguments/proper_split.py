ModelParams = dict(
    # Use stride=10 to select every 10th frame as test frame
    # According to paper: "We select every 10th frame from the sequences as the test frames and use the remaining frames for training"
    stride = 10,
)

OptimizationParams = dict(
    # Use a reasonable number of iterations for demonstration
    coarse_iterations = 5000,
    iterations = 50000,
) 