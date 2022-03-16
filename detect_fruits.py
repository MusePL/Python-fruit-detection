import cv2
import json
import click
import numpy as np
from glob import glob
from tqdm import tqdm

from typing import Dict


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str

        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    def count_bananas(image: np.ndarray, contour_length_bananas):
        structuring_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        result = cv2.inRange(image, (23, 65, 0), (44, 255, 255))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, structuring_element, iterations=5)

        contours, hierarchy = cv2.findContours(result, mode=cv2.CHAIN_APPROX_SIMPLE, method=cv2.RETR_TREE)

        bananas = 0
        for contour in contours:
            if cv2.contourArea(contour) > contour_length_bananas:
                bananas += 1

        return bananas

    def count_apples(image: np.ndarray, contour_length_apples):
        structuring_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        result = cv2.inRange(image, (0, 23, 45), (10, 220, 255))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, structuring_element, iterations=5)

        contours, hierarchy = cv2.findContours(result, mode=cv2.CHAIN_APPROX_SIMPLE, method=cv2.RETR_TREE)

        apples = 0
        for contour in contours:
            if cv2.contourArea(contour) > contour_length_apples:
                apples += 1

        return apples

    def count_oranges(image: np.ndarray, contour_length_oranges):
        structuring_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        result = cv2.inRange(image, (8, 61, 194), (21, 255, 247))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, structuring_element, iterations=5)

        contours, hierarchy = cv2.findContours(result, mode=cv2.CHAIN_APPROX_SIMPLE, method=cv2.RETR_TREE)

        oranges = 0
        for contour in contours:
            if cv2.contourArea(contour) > contour_length_oranges:
                oranges += 1

        return oranges

    contour_length_bananas = (img.shape[0] * img.shape[1]) / 1200
    contour_length_apples = (img.shape[0] * img.shape[1]) / 3000
    contour_length_oranges = (img.shape[0] * img.shape[1]) / 1500
    img = cv2.resize(img, dsize=None, fx=0.3, fy=0.3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    banana = count_bananas(img, contour_length_bananas)
    apple = count_apples(img, contour_length_apples)
    orange = count_oranges(img, contour_length_oranges)

    print(apple, banana, orange)
    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory')
@click.option('-o', '--output_file_path', help='Path to output file')
def main(data_path, output_file_path):

    img_list = glob(f'{data_path}/*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(img_path)

        filename = img_path.split('/')[-1]

        results[filename] = fruits
    
    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
