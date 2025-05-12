import cv2
import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import defaultdict
from detect_corners import find_corners


cfg = {
    "RESIZE_IMAGE_WIDTH": 1200,
    "EDGE_DETECTION": {
        "LOW_THRESHOLD": 90,
        "HIGH_THRESHOLD": 400,
        "APERTURE": 3,
        "method": "canny",
        "min_val": 50,
        "max_val": 150,
        "kernel_size": 3,
    },
    "LINE_DETECTION": {
        "THRESHOLD": 150,
        "DIAGONAL_LINE_ELIMINATION": True,
        "DIAGONAL_LINE_ELIMINATION_THRESHOLD_DEGREES": 30,
    },
    "RANSAC": {
        "OFFSET_TOLERANCE": 0.1,
        "BEST_SOLUTION_TOLERANCE": 0.15,
        "reproj_threshold": 3,
        "max_iters": 1000,
    },
    "MAX_OUTLIER_INTERSECTION_POINT_RATIO_PER_LINE": 0.7,
    "BORDER_REFINEMENT": {
        "SOBEL_KERNEL_SIZE": 3,
        "LINE_WIDTH": 4,
        "NUM_SURROUNDING_SQUARES_IN_WARPED_IMG": 5,
        "WARPED_SQUARE_SIZE": [50, 50],
        "EDGE_DETECTION": {
            "HORIZONTAL": {
                "APERTURE": 3,
                "HIGH_THRESHOLD": 300,
                "LOW_THRESHOLD": 120,
            },
            "VERTICAL": {
                "APERTURE": 3,
                "HIGH_THRESHOLD": 200,
                "LOW_THRESHOLD": 100,
            },
        },
    },
}

CATEGORY_ID_TO_SYMBOL = {
    0: "white_pawn",  # White pawn
    1: "white_rook",  # White rook
    2: "white_knight",  # White knight
    3: "white_bishop",  # White bishop
    4: "white_queen",  # White queen
    5: "white_king",  # White king
    6: "black_pawn",  # Black pawn
    7: "black_rook",  # Black rook
    8: "black_knight",  # Black knight
    9: "black_bishop",  # Black bishop
    10: "black_queen",  # Black queen
    11: "black_king",  # Black king
}


def warp_board(original_img, corners):
    (tl, tr, br, bl) = corners

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    if maxWidth > 4000 or maxHeight > 4000:
        return None
        # raise ValueError(
        #     f"Suspend warp: suspiciously large dimensions ({maxWidth}x{maxHeight})"
        # )
    if maxWidth < 100 or maxHeight < 100:
        return None
        # raise ValueError(f"Invalid warp: too small dimensions ({maxWidth}x{maxHeight})")

    side_length = min(original_img.shape[:2])  # ensure fits in image
    side_length = min(side_length, max(maxWidth, maxHeight))

    destination = np.array(
        [
            [0, 0],
            [side_length - 1, 0],
            [side_length - 1, side_length - 1],
            [0, side_length - 1],
        ],
        dtype="float32",
    )

    corners = np.array(corners, dtype=np.float32).reshape((4, 2))
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(original_img, matrix, (side_length, side_length))

    return warped


def split_board_into_squares(warped_img):
    squares = []
    h, w = warped_img.shape[:2]
    square_size_x = w / 8
    square_size_y = h / 8

    for y in range(8):
        for x in range(8):
            x1 = int(round(x * square_size_x))
            y1 = int(round(y * square_size_y))
            x2 = int(round((x + 1) * square_size_x))
            y2 = int(round((y + 1) * square_size_y))

            if x2 > w or y2 > h:
                continue  # avoid going out of bounds

            square = warped_img[y1:y2, x1:x2]
            squares.append(square)

    return squares


def complete_board_labels(piece_dict):
    files = "abcdefgh"
    ranks = "12345678"
    full_labels = {}

    for r in ranks:
        for f in files:
            square = f + r
            label = piece_dict.get(square, "empty")
            full_labels[square] = label

    return full_labels


def load_chessred_images(root_dir="chessred2k"):
    # images = []
    paths = []

    root_dir = os.path.join(root_dir, "images")

    for subdir in os.listdir(root_dir):
        full_subdir = os.path.join(root_dir, subdir)
        if os.path.isdir(full_subdir):
            for image_path in glob(os.path.join(full_subdir, "*.jpg")):
                # img = cv2.imread(image_path)
                # if img is not None:
                # images.append(img)
                paths.append(image_path)
                # else:
                #     print(f"Failed to load: {image_path}")

    return paths


def load_annotations(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def build_board_lookup(annotation_data):
    board_by_image_id = defaultdict(dict)  # {image_id: {square: piece}}

    for piece in annotation_data["annotations"]["pieces"]:
        image_id = piece["image_id"]
        category_id = piece["category_id"]
        position = piece["chessboard_position"]  # like "e4"

        symbol = CATEGORY_ID_TO_SYMBOL.get(category_id, "?")
        board_by_image_id[image_id][position] = symbol

    return board_by_image_id


def get_board_for_image(image_file_name, annotation_data, board_lookup):
    # Find the image_id corresponding to the image file name
    for image in annotation_data["images"]:
        if image["file_name"] == image_file_name:
            return board_lookup.get(image["id"], {})
    return {}


def rotate_image(image, angle):
    """Rotate image by a specified angle (in degrees)."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h))
    return rotated


def correct_board_orientation(warped_board, square_threshold=127):
    """
    Rotate the warped chessboard image so that the side with white pieces faces down.

    Args:
        warped_board (np.ndarray): Warped top-down view of the chessboard (grayscale or BGR image).

    Returns:
        np.ndarray: Possibly rotated board (white side facing down).
    """

    # Convert to grayscale if needed
    if warped_board.ndim == 3:
        gray = cv2.cvtColor(warped_board, cv2.COLOR_BGR2GRAY)
    else:
        gray = warped_board.copy()

    # Board size (assumed square)
    height, width = gray.shape
    square_h = height // 8
    square_w = width // 8

    # Sample the bottom-left square
    bottom_left = gray[-square_h:, :square_w]
    avg_brightness = np.mean(bottom_left)

    if avg_brightness >= square_threshold:
        # Square is white — rotate 90° clockwise
        warped_board = cv2.rotate(warped_board, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    # Get top 2 ranks and bottom 2 ranks
    top_rows = gray[0 : 2 * square_h, :]
    bottom_rows = gray[-2 * square_h :, :]

    # Compute average brightness
    avg_top = np.mean(top_rows)
    avg_bottom = np.mean(bottom_rows)

    # Rotate if top is brighter (assume white pieces on top)
    if avg_top > avg_bottom:
        # print("Flipping board to place white side at bottom")
        warped_board = cv2.rotate(warped_board, cv2.ROTATE_180)

    return warped_board


def get_square_keys_in_order():
    return [x + str(y) for y in range(8, 0, -1) for x in "abcdefgh"]


# Test functions
def test_chessboard_detection(image_path):
    # Step 1: Load original image
    original = cv2.imread(image_path)

    # Show original image
    plt.figure(figsize=(8, 8))
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Step 3: Use Wolflein's corner detection
    corners = find_corners(cfg, original)

    if corners is None or len(corners) != 4:
        print("Failed to detect corners.")
        return None

    print("Detected corners:", corners)

    # Draw detected corners
    corners_img = original.copy()
    for pt in corners:
        pt = tuple(map(int, pt))
        cv2.circle(corners_img, pt, 10, (0, 0, 255), -1)

    plt.figure(figsize=(8, 8))
    plt.title("Detected Corners")
    plt.imshow(cv2.cvtColor(corners_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Step 4: Warp board to top-down view
    warped = warp_board(original, corners)

    corrected = correct_board_orientation(warped)

    # Step 5: Split warped board into squares
    squares = split_board_into_squares(corrected)

    # Step 6: Plot warped board
    plt.figure(figsize=(8, 8))
    plt.title("Warped Board")
    plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Step 6: Plot warped board
    plt.figure(figsize=(8, 8))
    plt.title("Corrected Board")
    plt.imshow(cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # Step 7: Plot first 16 squares
    plt.figure(figsize=(10, 10))
    for i in range(min(16, len(squares))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(cv2.cvtColor(squares[i], cv2.COLOR_BGR2RGB))
        plt.axis("off")
    plt.suptitle("First 16 Squares")
    plt.show()

    return squares


if __name__ == "__main__":
    squares = test_chessboard_detection("G000_IMG037.jpg")
