# Sudoku Solver using Convolutional Neural Networks (CNN)

This repository contains code for Solving  a Sudoku puzzle image using a Convolutional Neural Network (CNN) implemented in PyTorch. 
The project objective is to train a deep learning model that can solve a Sudoku puzzle. To help you, we have split the problem into two tasks: in Task 1 you shall train a model to read the Sudoku puzzle to know which digits there are and where are their placed within the Sudoku grid. In Task 2 you shall train a model to actually solve the Sudoku (using the standard rules), that is to complete the empty cells in the grid with the correct digits.

I have train a model that can complete the Sudoku puzzle from the partially filled in images. Please refer to Task 1 for the basic description of the Sudoku images.

To train such a model I have provide you with a training set of 50000 pairs of partially and completely filled in Sudoku images.

My task is to complete or to solve the 10000 test partially filled Sudoku images.


## Contents
1. **Data Loading and Preprocessing:**

The code begins by importing necessary libraries including PyTorch, pandas, numpy, and random.
It loads the Sudoku puzzle data from a CSV file. The CSV contains two columns: "Quiz" (representing the unsolved Sudoku puzzle) and "Solution" (representing the solved puzzle).


2. **Dataset Creation:**

A custom SudokuDataset class is defined to handle the dataset.
This class inherits from PyTorch's Dataset class and preprocesses the Sudoku puzzles and solutions for training.
It includes methods to retrieve puzzle-solution pairs, handle data augmentation, and convert data into appropriate tensor formats.

3. **Model Architecture:**

The code defines a CNN model called SudokuSolver using PyTorch's nn.Module class.
The model consists of convolutional layers, batch normalization, dropout layers, and fully connected layers.
The CNN model aims to predict the digits for each cell in the Sudoku puzzle grid.

4. **Training:**

The model is trained using the training dataset.
Training is performed using a custom training loop.
The RMSprop optimizer is used for optimizing model parameters, and cross-entropy loss is used as the loss function.
During training, metrics such as loss and accuracy are computed and displayed for each epoch.

5.**Evaluation:**

After training, the model is evaluated on the validation dataset to assess its performance.
Loss and accuracy metrics are computed for both training and validation sets.

6. **Test Set Prediction:**

Once trained, the model is used to predict solutions for Sudoku puzzles in the test dataset.
Predictions are generated for each puzzle and converted into the required format.
The predicted solutions are saved to a CSV file (Test_Predictions.csv)


## Features

- **Data Loading and Preprocessing**: The project loads Sudoku puzzle data from a CSV file containing both the unsolved puzzles and their solutions. The data is preprocessed and converted into a suitable format for training.

- **Dataset Creation**: A custom `SudokuDataset` class is implemented to handle the dataset. It preprocesses the puzzles and solutions, handles data augmentation, and converts the data into appropriate tensor formats.

- **Model Architecture**: The project defines a CNN model called `SudokuSolver` using PyTorch's `nn.Module` class. The model architecture consists of convolutional layers, batch normalization, dropout layers, and fully connected layers.

- **Training and Evaluation**: The model is trained using the training dataset with the RMSprop optimizer and cross-entropy loss. Training progress is monitored, and evaluation metrics such as loss and accuracy are computed for both training and validation sets.

- **Test Set Prediction**: Once trained, the model is used to predict solutions for Sudoku puzzles in the test dataset. The predicted solutions are saved to a CSV file for further analysis.


## Files Included
- `task-2.ipynb`: Main Python script containing the code for training the CNN model, loading test images, and generating the output CSV file.
- `Training_Dataset.csv` : This CSV file containing the Quiz and solution of 50000 Training datasets.
- `Test_Dataset.csv` : This CSV file containing the predicted digits of the 10000 test Images of sudoku(Unsolved Sudoku With labels).
- `Test_Predictions.csv` : This CSV file containing the Solved Sudoku Puzzles of 10000 Images.
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
