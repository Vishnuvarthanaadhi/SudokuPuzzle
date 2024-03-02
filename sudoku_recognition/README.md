# Sudoku Digit Recognition using Convolutional Neural Networks (CNN)

This repository contains code for recognizing digits in a Sudoku puzzle image using a Convolutional Neural Network (CNN) implemented in PyTorch. The CNN is trained on the MNIST dataset, which consists of handwritten digits. After training, the model is used to predict digits in Sudoku puzzle images.
The project objective is to train a deep learning model that can solve a Sudoku puzzle. To help you, we have split the problem into two tasks: in Task 1( **Sudoku_Recognition**) you shall train a model to read the Sudoku puzzle to know which digits there are and where are their placed within the Sudoku grid. 

In Task 2( **Sudoku_solve**) you shall train a model to actually solve the Sudoku (using the standard rules), that is to complete the empty cells in the grid with the correct digits.

The Sudoku puzzle is a 9x9 grid filled with digits from 1 to 9 following the Sudoku rules (each column, each row, and each 3x3 subgrid contains all of the digits). The partially completed grid that you shall be able to read here has always 40 cells empty. My model shall be able to recognize the filled digits within the puzzle and their positions (cells) in the grid. In the end, it shall provide a list of all the cells with the digits within.
My task is to recognize the digits and their positions for all 10000 test set images we provide.

**Evaluation**

The goal of this task is to predict the correct label for each digit in all the test images. In addition to the digit class, I will also submit each digit's cell position in the puzzle.

The cell position within the 9x9 grid is described by its x (horizontal) and y (vertical) coordinates similarly as in a matrix. The indexes range from 0 to 8 such that top-left is (0, 0), top-right is (0, 8), bottom-left is (8, 0) and bottom-right is (8, 8).

**Submission File**
Each test image is identified with an ID (0-9999). For each ID you will need to submit 81 values (for each cell within the 9x9 grid). You shall use the value (digit) 0 for an empty cell. The submission file should contain a header and have the following format:

id,value

0_00,0

0_01,1

0_02,5

._.,.

._.,.

9999_88,9

Here id is composed of the ID of the image (0-9999) and the xy coordinate of the position ID_xy, value is the digit (1-9, with 0 used for empty positions). In total, there shall be 810001 lines in the submission file including the header.

## Contents
1. **Requirements**: 
   - MINIST Datasets, Test Image (which contains 10000 images of Sudoku puzzles with empty values).
2. **Dataset Preparation**: 
   - Downloads and preprocesses the MNIST dataset for training and testing the CNN.
3. **CNN Model**: 
   - Defines the architecture of the CNN model used for digit recognition. The model consists of two convolutional layers followed by two fully connected layers.
4. **Training**: 
   - Trains the CNN model using the MNIST dataset. The model is trained for a specified number of epochs, and the training loss is printed for each epoch.
5. **Test Image Loading**: 
   - Loads the test images containing Sudoku puzzles. The path to the test images is specified in the code.
6. **Digit Prediction Functions**: 
   - Defines functions to preprocess test image cells and predict digits using the trained CNN model. The preprocess_and_predict function preprocesses a single cell image and predicts the digit using the trained model.
7. **Digit Prediction and Combination**: 
   - Applies the prediction functions to predict digits for all cells in the Sudoku puzzles. It iterates through all the cells in the Sudoku puzzles, preprocesses each cell image, predicts the digit using the CNN model, and combines the predictions with their coordinates.
8. **Output Generation**: 
   - Generates a CSV file containing the predicted digits for submission. It iterates through all the predicted results for the test images, converts the coordinates and predicted digits to the required format, and writes them to a CSV file.

## Files Included
- `task-1.ipynb`: Main Python script containing the code for training the CNN model, loading test images, predicting digits, and generating the output CSV file.
- `test_recognition.npy` : This NPY file containing the 10000 test Images of sudoku with empty Values(Unsolved Puzzles).
- `Output_Task1.csv` : This CSV file containing the predicted digits of the 10000 test Images of sudoku.[Download Output_Task1](https://github.com/Vishnuvarthanaadhi/SudokuPuzzle/blob/8b98bbe13629c034817e08cd6a5b67f0c82b5042/sudoku_recognition/Output_Task1.csv)
- `README.md`: Markdown file explaining the code and usage instructions.

## Notes
- Ensure that you have GPU support if available for faster training and inference.
- Make sure the file paths for dataset loading, test image loading, and output generation are correctly specified.
- Feel free to adjust hyperparameters, model architecture, or preprocessing steps based on your requirements and experimentation results.

## Author
Vishnuvarthan Adhimoolam

## Acknowledgements
- This code is adapted from various tutorials and resources available online.
- The CNN architecture and training process are based on standard practices in deep learning.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
