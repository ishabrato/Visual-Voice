import cv2

def get_available_cameras():
    """
    Detects and returns a list of available camera indices.
    """
    available_cameras = []
    # Test up to 10 camera indices (0 to 9)
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to read a frame to confirm the camera is truly functional
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras

def get_camera_name(index):
    """
    Attempts to get a more descriptive name for a camera by its index.
    This is often system-dependent and might just return the index.
    """
    # This is a placeholder. Getting a true descriptive name is complex and OS-dependent.
    # For now, we'll just return "Camera X"
    return f"Camera {index}"

def load_camera_setting():
    """
    Loads the preferred camera index from a configuration file.
    """
    try:
        with open("camera_config.txt", "r") as f:
            index = int(f.read().strip())
            return index
    except (FileNotFoundError, ValueError):
        return 0 # Default to camera 0 if file not found or invalid

def save_camera_setting(index):
    """
    Saves the preferred camera index to a configuration file.
    """
    with open("camera_config.txt", "w") as f:
        f.write(str(index))
