import utils
import resnet18
import numpy as np
import detect_corners
import matplotlib.pyplot as plt
import cv2
import os
from torchvision import transforms
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import json

# Map piece names to integer labels
label_map = {
    "white_pawn": 0,
    "white_rook": 1,
    "white_knight": 2,
    "white_bishop": 3,
    "white_queen": 4,
    "white_king": 5,
    "black_pawn": 6,
    "black_rook": 7,
    "black_knight": 8,
    "black_bishop": 9,
    "black_queen": 10,
    "black_king": 11,
    "empty": 12,
}

label_inv_map = {
    0: "white_pawn",
    1: "white_rook",
    2: "white_knight",
    3: "white_bishop",
    4: "white_queen",
    5: "white_king",
    6: "black_pawn",
    7: "black_rook",
    8: "black_knight",
    9: "black_bishop",
    10: "black_queen",
    11: "black_king",
    12: "empty",
}


class ChessSquareDataset(Dataset):
    def __init__(self, image_paths, annotations, transform=None, max_attempts=4):
        self.image_paths = image_paths
        self.annotations = annotations
        self.board_lookup = utils.build_board_lookup(annotations)
        self.transform = transform
        self.max_attempts = max_attempts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Generate (square image, label) pairs across all images
        self.samples = []
        i = 0

        # num_workers = 3 # Use N-1 cores
        # with ProcessPoolExecutor(max_workers=num_workers) as executor:
        #     futures = [
        #         executor.submit(
        #             process_image,
        #             path,
        #             self.annotations,
        #             self.board_lookup,
        #             self.transform,
        #             self.max_attempts,
        #         )
        #         for path in self.image_paths
        #     ]

        # for f in as_completed(futures):
        #     result = f.result()
        #     if result:
        #         self.samples.extend(result)

        self.samples = safe_parallel_load(
            self.image_paths, self.annotations, self.board_lookup, self.transform
        )

        # for img_path in image_paths:
        #     # try:
        #     # print(f"i{i}")
        #     i += 1
        #     board_samples = self._process_image(img_path)
        #     if board_samples is None:
        #         continue
        #     self.samples.extend(board_samples)
        #     # except Exception as e:
        # print(f"[Warning] Skipping {img_path}: {e}")

    def _process_image(self, img_path):
        # Load and rotate image to find corners
        image = cv2.imread(img_path)
        if image is None:
            return None
            # raise ValueError(f"Image at path {img_path} not found or unreadable")

        attempt = 0
        while attempt < self.max_attempts:
            corners = detect_corners.find_corners(utils.cfg, image)

            if corners is None or len(corners) != 4:
                attempt += 1
                angle = -22.5 + attempt * 15
                image = utils.rotate_image(image, angle)
            else:
                break
        else:
            return None
            # raise RuntimeError(
            #     f"Corner detection failed after {self.max_attempts} attempts"
            # )

        # Warp and split board
        warped = utils.warp_board(image, corners)
        if warped is None:
            return None
        warped = utils.correct_board_orientation(warped)
        squares = utils.split_board_into_squares(
            warped
        )  # Returns list of 64 square images

        # Get labels
        image_name = os.path.basename(img_path)
        board = utils.get_board_for_image(
            image_name, self.annotations, self.board_lookup
        )
        square_labels = utils.complete_board_labels(
            board
        )  # Dict like {'a1': 'white_rook', ...}

        # Square order: a8 to h8, ..., a1 to h1 (top to bottom, left to right)
        square_keys = utils.get_square_keys_in_order()  # Assumes top-left is a8

        image_label_pairs = []
        for square_img, key in zip(squares, square_keys):
            label_name = square_labels.get(key, "empty")
            label = label_map[label_name]
            if self.transform:
                square_img = self.transform(square_img)
            else:
                square_img = transforms.ToTensor()(square_img)
            image_label_pairs.append((square_img, label))

        return image_label_pairs

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def save_squares_and_labels(output_root, image_name, squares, labels):
    """
    Saves each square image to a folder and a labels.json file.
    """
    board_dir = os.path.join(output_root, image_name.split(".")[0])
    os.makedirs(board_dir, exist_ok=True)

    label_dict = {}
    square_keys = utils.get_square_keys_in_order()  # a1 to h8

    for sq_img, key, label_id in zip(squares, square_keys, labels):
        # Save image as .jpg (denormalize if using ToTensor)
        if isinstance(sq_img, torch.Tensor):
            sq_img = sq_img.mul(255).byte().permute(1, 2, 0).cpu().numpy()
        img_path = os.path.join(board_dir, f"{key}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(sq_img, cv2.COLOR_RGB2BGR))
        label_dict[key] = int(label_id)  # Save int label

    # Save label mapping
    with open(os.path.join(board_dir, "labels.json"), "w") as f:
        json.dump(label_dict, f)


def safe_parallel_load(paths, annotations, board_lookup, transform):
    from functools import partial
    from tqdm import tqdm

    process_fn = partial(
        process_image,
        annotations=annotations,
        board_lookup=board_lookup,
        transform=transform,
    )

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_fn, path) for path in paths]
        for f in tqdm(futures):
            try:
                res = f.result()
                if res:
                    results.extend(res)
            except Exception as e:
                print(f"Warning: {e}")

    return results


def process_image(img_path, annotations, board_lookup, transform, max_attempts=4):
    
    image_name = os.path.basename(img_path)
    board_dir = os.path.join("dataset_cached", image_name.split(".")[0])
    label_path = os.path.join(board_dir, "labels.json")

    if os.path.exists(label_path):
        # Load preprocessed labels
        try:
            with open(label_path, "r") as f:
                label_dict = json.load(f)
            square_keys = utils.get_square_keys_in_order()
            image_label_pairs = []
            for key in square_keys:
                img_path = os.path.join(board_dir, f"{key}.jpg")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"{img_path} missing.")
                square_img = cv2.imread(img_path)
                square_img = cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB)
                if transform:
                    square_img = transform(square_img)
                else:
                    square_img = transforms.ToTensor()(square_img)
                image_label_pairs.append((square_img, int(label_dict.get(key, 12))))  # default to empty
            return image_label_pairs
        except Exception as e:
            print(f"[Warning] Failed to load cached data for {image_name}: {e}")   
    
    
    image = cv2.imread(img_path)
    if image is None:
        return None

    attempt = 0
    while attempt < max_attempts:
        corners = detect_corners.find_corners(utils.cfg, image)
        if corners is None or len(corners) != 4:
            attempt += 1
            angle = -22.5 + attempt * 15
            image = utils.rotate_image(image, angle)
        else:
            break
    else:
        return None

    warped = utils.warp_board(image, corners)
    if warped is None:
        return None
    warped = utils.correct_board_orientation(warped)
    squares = utils.split_board_into_squares(warped)

    # image_name = os.path.basename(img_path)
    board = utils.get_board_for_image(image_name, annotations, board_lookup)
    square_labels = utils.complete_board_labels(board)
    square_keys = utils.get_square_keys_in_order()

    labels = []
    image_label_pairs = []
    for square_img, key in zip(squares, square_keys):
        label_name = square_labels.get(key, "empty")
        label = label_map[label_name]
        labels.append(label)
        if transform:
            square_img = transform(square_img)
        else:
            square_img = transforms.ToTensor()(square_img)
        image_label_pairs.append((square_img, label))

    save_squares_and_labels("dataset_cached", image_name, squares, labels)

    return image_label_pairs


def main():
    # Load image paths and annotations
    paths = utils.load_chessred_images()
    annotations = utils.load_annotations("annotations.json")
    board_lookup = utils.build_board_lookup(annotations)
    # paths = paths[:50]

    # Split dataset
    train_paths, temp_paths = train_test_split(paths, test_size=0.3, random_state=42)
    val_paths, test_paths = train_test_split(temp_paths, test_size=0.5, random_state=42)

    # Define transform and device
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 3: Create datasets and loaders
    train_dataset = ChessSquareDataset(train_paths, annotations, transform=transform)
    val_dataset = ChessSquareDataset(val_paths, annotations, transform=transform)
    test_dataset = ChessSquareDataset(test_paths, annotations, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize model, loss, optimizer
    model = resnet18.ResNet18(num_classes=13).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train model
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Train Loss = {avg_loss:.4f}")

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        print(f"Epoch {epoch + 1}: Validation Loss = {val_loss / len(val_loader):.4f}")

    # Save Model
    # model_path = "resnet18_chess_model.pt"
    # torch.save(model.state_dict(), model_path)
    # print(f"Model saved to {model_path}")

    # Load Model
    # model = resnet18.ResNet18(num_classes=13)
    # model.load_state_dict(torch.load("resnet18_chess_model.pt"))
    # model.eval()

    accuracies = []
    perfect_boards = 0
    for path in test_paths:
        image = cv2.imread(path)
        image_name = os.path.basename(path)
        true_board = utils.get_board_for_image(image_name, annotations, board_lookup)
        pred_board = predict_board(
            model, image, annotations, device, transform, label_inv_map
        )
        if pred_board is None:
            continue
        acc, is_perfect = evaluate_board_accuracy(pred_board, true_board)
        print(f"{image_name}: Accuracy = {acc:.2%}")
        accuracies.append(acc)
        perfect_boards += int(is_perfect)

    print(f"Average board accuracy: {np.mean(accuracies):.2%}")
    print(
        f"Perfect Boards (All 64 squares correct): {perfect_boards}/{len(accuracies)} = {100 * perfect_boards / len(accuracies):.2f}%"
    )


def predict_board(model, image, annotations, device, transform, label_inv_map):
    model.eval()
    with torch.no_grad():
        attempt = 0
        while attempt < 4:
            corners = detect_corners.find_corners(utils.cfg, image)
            if corners is None or len(corners) != 4:
                attempt += 1
                angle = -22.5 + attempt * 15
                image = utils.rotate_image(image, angle)
            else:
                break
        else:
            # raise RuntimeError("Corner detection failed after 10 attempts")
            return None

        warped = utils.warp_board(image, corners)
        if warped is None:
            return None
        warped = utils.correct_board_orientation(warped)
        squares = utils.split_board_into_squares(warped)

        predicted_labels = []
        for sq in squares:
            tensor = transform(sq).unsqueeze(0).to(device)
            output = model(tensor)
            pred_class = torch.argmax(output, dim=1).item()
            predicted_labels.append(label_inv_map[pred_class])  # e.g., "white_queen"

        # Construct board dictionary
        square_keys = utils.get_square_keys_in_order()
        predicted_board = dict(zip(square_keys, predicted_labels))

        return predicted_board


def evaluate_board_accuracy(predicted_board, true_board):
    total = 0
    correct = 0
    all_match = True
    for square, label in predicted_board.items():
        if square not in true_board:
            continue
        total += 1
        if label == true_board[square]:
            correct += 1
        else:
            all_match = False
    acc = correct / total if total > 0 else 0
    return acc, all_match


main()
