A Java-based neural network for image classification and digit recognition using CSV, IDX (MNIST), and JPG files. Built with custom Java classes for complete transparency, educational value, and my own pleasure.

Features
Custom Neural Network implementation (train and test from scratch)

Support for multiple input types: CSV, IDX (MNIST format), and JPG images

Persistent model saving/loading for weights and biases

Java Swing GUI for easy interaction (training, testing, file selection)

Modifiable for experimentation, educational projects, or practical use

Demo
<img width="806" height="402" alt="Screenshot 2025-09-08 at 9 50 26 PM" src="https://github.com/user-attachments/assets/f189c666-c7fa-4474-bc33-0051efe39dee" />


Usage

Project Structure

How it Works

Customization

Contributing

License

Getting Started
Prerequisites
Java 8 or higher

IDE (IntelliJ IDEA, Eclipse, NetBeans, or command line)

MNIST dataset files (or your own images)

Installation
bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
Open the project in your Java IDE.
All dependencies are standard Java libraries.

Usage
Training:

Use the GUI buttons to train on CSV or IDX MNIST files.

Input the desired training sample count.

Prediction:

Select modes to test on a CSV row, IDX row, or JPG image (converted and normalized internally).

Results display in the GUI.

Saving/Loading:

Weights and biases auto-save after training and are loaded for prediction.

Project Structure
File/Class	Purpose
Main.java	GUI, flow control, training/testing options
NeuralNetwork.java	Core network logic, training, forward propagation, weight management
Neuron.java	Individual neuron logic, weights, bias, activation, leaky ReLU
FileManager.java	CSV/JPG reading/writing, model persisting, parsing utilities
IDX3Reader.java	Handles IDX (MNIST) file reading and normalization
MnistImageProcessor.java	Reads and normalizes raw MNIST images
How it Works
Model: 2 hidden layers and 1 output layer, Leaky ReLU activations, He initialization

Training: Supports batch learning via backpropagation and weight updates

Inputs: Accepts images as arrays, normalizing as needed

Persistence: Model weights/biases saved in CSV for transparency/easy edit

Customization
Edit neural network parameters in Main.java (hiddenLayerNeurons, epochs, learningRate)

Plug in new data formats by expanding utility classes

Experiment with model architecture (layers, activation, etc.)

Contributing
Contributions and forks are welcome! Please open issues or pull requests for bugfixes, improvements, or new features.

License
MIT or specify your own.

TODOs
Add sample trained model and test images (optionally as releases/assets)

Write unit tests for key functions

Optional: Dockerfile for reproducible builds

Feel free to copy, edit, and expand this README for your project specifics, screenshots, and contributor guidelines! If you want it personalized further (project name, screenshot/image, or more setup detail), let me know.
