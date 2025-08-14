ModelParams = dict(
    # Maintain the same data split as stage 1
    stride = 10,
    # Configure time range for phase 2
    original_start_time = 0,
    start_time = 50,
    end_time = 99,
)

OptimizationParams = dict(
    # Use a reasonable number of iterations for demonstration
    coarse_iterations = 5000,
    iterations = 50000,
) 