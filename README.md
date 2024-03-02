# Sudoku Solver using Deep Learning

## Overview

This project aims to develop a deep learning model capable of solving Sudoku puzzles. To achieve this, we have divided the problem into two main tasks:

1. **Task 1: Puzzle Recognition or Hand Written Digit Recognition**
   - In this task, the objective is to train a model to recognize and understand the layout of a Sudoku puzzle. The model should be able to identify the digits present in the puzzle and their respective positions within the grid.

2. **Task 2: Puzzle Solving**
   - Once the puzzle is recognized, the next task is to train a model that can solve the Sudoku puzzle using standard rules. The model should be capable of filling in the empty cells of the grid with the correct digits to complete the puzzle.

## Repository Structure

- **/task1**: Contains code and resources related to Task 1, including data preprocessing, model training, and evaluation for puzzle recognition.
- **/task2**: Includes code and resources for Task 2, focusing on puzzle solving using deep learning techniques.
- **/data**: This directory holds the datasets used for training and testing the models.
- **/models**: Stores trained models generated during the development process.
- **/utils**: Utility scripts or functions used across tasks, such as data loading and preprocessing.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the respective task directory (`task1` or `task2`) based on your current focus.
3. Follow the instructions provided in the README files within each task directory to set up the environment, train the models, and evaluate their performance.

## Dependencies

Make sure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib (for visualization, if needed)

## Contributing

Contributions to this project are welcome! If you have any ideas for improvements, bug fixes, or new features, feel free to open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE), allowing for free usage, modification, and distribution of the codebase.
