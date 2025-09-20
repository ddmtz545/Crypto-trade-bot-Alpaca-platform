import time

def start_timer():
    """
    Starts a high-precision timer.
    Equivalent to MATLAB's 'tic'.

    Returns:
        float: The start time from the performance counter.
    """
    print("--- Timer Started ---")
    return time.perf_counter()

def end_timer(start_time):
    """
    Stops the timer and prints the elapsed time since the start time.
    Equivalent to MATLAB's 'toc'.

    Args:
        start_time (float): The time returned by start_timer().
    """
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print("\n-------------------------------------------------")
    print(f"Total script execution time: {elapsed_time:.2f} seconds")
    print("-------------------------------------------------")
    return elapsed_time
