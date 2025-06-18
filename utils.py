
import matplotlib.pyplot as plt
import numpy as np
import os

from batch_loader import generate_gril


def visualise(image: np.ndarray, gaze: np.ndarray) -> None: 
    #image: A 3D NumPy array of shape (H, W, 3) (RGB image) - (height, weight , no. of channels)
    #gaze: A 2D point [x, y] in normalized coordinates (range 0–1).

    """
        shows the image with its gaze point
    """

    gaze = gaze * [image.shape[0], image.shape[1]] #Scales the gaze point to pixel coordinates by multiplying with image dimensions

    plt.imshow(image)
    plt.scatter(gaze[0], gaze[1], c='red', s=40, label='Gaze') #Displays the image and plots a red dot where the person looked.
    plt.title(f"Gaze at: {gaze}")
    plt.legend()
    plt.axis('off')
    plt.show() #Adds a title showing the gaze position, turns off axis, and shows the plot.


def test_generate_gril(path: str, file_list: list, max: int):
    """
        function to test generate_gril() 
    """

    gen = generate_gril(path, file_list) #Calls the generator function to yield samples.

    for i, (x, y) in enumerate(gen):
        print(f"Sample : {i+1}")
        print(f"Image : {x['image'].shape}")
        print(f"Depth : {x['depth'].shape}")
        print(f"Gaze : {y['gaze']}")
        print(f"Action : {y['action']}")
        print("-"*15)

        if i == max:
            break


def total_samples_count(path: str, file_list: list) -> int:
    """
        to get total count of samples 
    """
    total = 0

    for file in file_list:
        data = np.load(os.path.join(path, file))
        total += data["images"].shape[0]

    return total
