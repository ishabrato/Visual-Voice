import cv2

def find_supported_resolutions():
    """
    Tests a list of common resolutions to see which ones are supported by the camera.
    """
    supported_resolutions = []
    # List of common resolutions to test (width, height)
    resolutions_to_test = [
        (1920, 1080),
        (1280, 720),
        (800, 600),
        (640, 480),
        (320, 240),
    ]

    print("Testing camera resolutions...")

    for width, height in resolutions_to_test:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            break

        # Set the desired resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Read the actual resolution set by the camera
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if the camera accepted the resolution
        if actual_width == width and actual_height == height:
            supported_resolutions.append((width, height))
            print(f"  - Supported: {width}x{height}")
        else:
            print(f"  - Not Supported: {width}x{height} (Camera returned {actual_width}x{actual_height})")

        cap.release()

    if not supported_resolutions:
        print("\nCould not find any supported resolutions from the test list.")
        # Try to get the default resolution
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            default_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Default camera resolution is: {default_width}x{default_height}")
            supported_resolutions.append((default_width, default_height))
            cap.release()
    else:
        print(f"\nFound supported resolutions: {supported_resolutions}")

    return supported_resolutions

if __name__ == "__main__":
    find_supported_resolutions()
