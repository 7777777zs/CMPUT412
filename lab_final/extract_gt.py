#!/usr/bin/env python3

import cv2
import numpy as np
import argparse

IMAGE = "/home/antonio/Pictures/Screenshots/Screenshot from 2025-04-20 14-15-08.png"


#!/usr/bin/env python3


def setup_blob_detector(min_area=20, max_area=2000, min_dist_between_blobs=5):
    """Configure and create a blob detector optimized for circular pattern detection"""
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 220
    params.thresholdStep = 10

    # Filter by Area
    params.filterByArea = True
    params.minArea = min_area
    params.maxArea = max_area

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    # Minimum distance between blobs
    params.minDistBetweenBlobs = min_dist_between_blobs

    # Create detector
    return cv2.SimpleBlobDetector_create(params)


def detect_board(image, pattern_dims=(7, 3), blob_detector=None):
    """
    Detect circle grid in image and return detection status and centers.

    Args:
        image (np.ndarray): Input image
        pattern_dims (tuple): Circle pattern dimensions (width, height)
        blob_detector: OpenCV blob detector

    Returns:
        tuple: (detection status, centers of detected blobs)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Find circle grid
    detection, centers = cv2.findCirclesGrid(
        gray,
        patternSize=pattern_dims,
        flags=cv2.CALIB_CB_SYMMETRIC_GRID,
        blobDetector=blob_detector,
    )

    return detection, centers


def get_board_center(centers, image_shape):
    """
    Calculate board center coordinates normalized to [-1, 1].

    Args:
        centers: Centers of detected blobs
        image_shape: Shape of the image (height, width)

    Returns:
        tuple: (center_x_normalized, center_y_normalized)
    """
    if centers is None or len(centers) == 0:
        return 0.0, 0.0

    # Get image dimensions
    img_height, img_width = image_shape[:2]

    # Extract center points
    center_points = [p[0] for p in centers]

    # Calculate board center (centroid of all blob centers)
    center_x = np.mean([p[0] for p in center_points])
    center_y = np.mean([p[1] for p in center_points])

    # Normalize center coordinates to range [-1, 1]
    # where (0,0) is the center of the image
    center_x_normalized = (center_x - img_width/2) / (img_width/2)
    center_y_normalized = (center_y - img_height/2) / (img_height/2)

    return center_x_normalized, center_y_normalized


def main():
    parser = argparse.ArgumentParser(
        description='Detect circular grid pattern in an image')
    parser.add_argument('image_path', help='Path to the input image')
    parser.add_argument('--pattern_width', type=int,
                        default=7, help='Width of the circle pattern')
    parser.add_argument('--pattern_height', type=int,
                        default=3, help='Height of the circle pattern')
    parser.add_argument('--min_area', type=int, default=20,
                        help='Minimum blob area')
    parser.add_argument('--max_area', type=int,
                        default=2000, help='Maximum blob area')
    parser.add_argument('--min_dist', type=int, default=5,
                        help='Minimum distance between blobs')
    parser.add_argument('--display', action='store_true',
                        help='Display the image with detected blobs')

    args = parser.parse_args()

    # Read the image
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not read image from {args.image_path}")
        return

    # Create blob detector
    blob_detector = setup_blob_detector(
        min_area=args.min_area,
        max_area=args.max_area,
        min_dist_between_blobs=args.min_dist
    )

    # Detect the board
    pattern_dims = (args.pattern_width, args.pattern_height)
    detection, centers = detect_board(image, pattern_dims, blob_detector)

    # Print detection info
    print(f"Image dimensions: {image.shape[1]}x{image.shape[0]}")
    print(f"Board detected: {detection}")

    if detection:
        # Calculate board center
        board_center_x, board_center_y = get_board_center(centers, image.shape)

        print(f"Number of dots detected: {len(centers)}")
        print(
            f"Board center (normalized): ({board_center_x:.3f}, {board_center_y:.3f})")

        # Raw center (in pixels)
        img_height, img_width = image.shape[:2]
        raw_center_x = (board_center_x * (img_width/2)) + (img_width/2)
        raw_center_y = (board_center_y * (img_height/2)) + (img_height/2)
        print(
            f"Board center (pixels): ({raw_center_x:.1f}, {raw_center_y:.1f})")

        # Print individual dot centers
        print("\nIndividual dot centers (pixels):")
        for i, point in enumerate(centers):
            x, y = point[0]
            print(f"Dot {i+1}: ({x:.1f}, {y:.1f})")

        # Draw dots and center on the image if display option is set
        if args.display:
            # Draw detected dots
            cv2.drawChessboardCorners(image, pattern_dims, centers, detection)

            # Draw board center
            center_x = int(raw_center_x)
            center_y = int(raw_center_y)
            cv2.circle(image, (center_x, center_y), 10, (0, 255, 0), -1)
            cv2.putText(image, "Board Center", (center_x+15, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the image
            cv2.imshow('Detected Dots', image)
            print("\nPress any key to close the image window")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("No circular grid pattern detected in the image")

        # Try to detect individual blobs and display info
        keypoints = blob_detector.detect(image)
        print(f"\nIndividual blobs detected: {len(keypoints)}")

        if len(keypoints) > 0:
            print("\nIndividual blob information:")
            for i, keypoint in enumerate(keypoints):
                x, y = keypoint.pt
                size = keypoint.size
                print(
                    f"Blob {i+1}: Position ({x:.1f}, {y:.1f}), Size: {size:.1f}")

            if args.display:
                # Draw keypoints on the image
                image_with_keypoints = cv2.drawKeypoints(
                    image, keypoints, np.array([]), (0, 0, 255),
                    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
                )
                cv2.imshow('Detected Blobs', image_with_keypoints)
                print("\nPress any key to close the image window")
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
