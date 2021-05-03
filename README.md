# Pytorch

PyTorch is a popular open-source machine learning library built on top of the Torch library. It is prominently used for deep learning. PyTorch serves primarily two purposes as a machine learning framework. First, it provides an interface to perform NumPy-like tensor operations using GPUs and CPUs, and second, the ability to set up machine learning algorithms that require iterative optimization (including deep learning). 


## Syllabus

- **Linear Algebra** - A good understanding of concepts such as vectors, matrices, tensors, matrix multiplication, etc.
- **Gradients and AutoGrad** - A theoretical understanding of gradients and how they are used to update parameters in machine learning models. Practical usage of torch, autograd and its features to update such parameters.
- **Using GPUs** - Usage of torch.device and .cuda() to leverage GPUs using PyTorch.
- **Tensor Operations on GPU** - Ability to create and manipulate tensors using PyTorch such as reshaping, multiplying, sum across different axes, transformations, etc. Ability to perform tasks such as converting an RGB image to black and white.
- **Dynamic Computational Graphs** - A theoretical understanding of computational graphs and the ability to visualize simple programs such as the addition of two numbers as computational graphs. An understanding of the difference between dynamic and static computational graphs.
- **Linear Regression** - Implementing linear regression with basic operations described in the competencies above.
- **Data loader** - Ability to implement custom data loaders and awareness of different arguments and features it provides. Loading large datasets onto memory is not efficient. 
- **Optimizers** - PyTorch provides a library of many different gradient optimizers such as Adam Optimizer, SGD, RMSProp, Adagrad, etc.
- **torch.NN** - A detailed understanding of this module which is the core of building computational graphs for machine learning. It contains a varied number of blocks such as convolutional layers, recurrent layers, transformer layers, loss functions, etc.
- **Saving and loading parameters** - Ability to store and load models or parameters.
- **Fully Connected Neural Networks** - Implementing the entire pipeline for a deep neural network (data loading, transformations, neural network layers (torch.nn), loss, optimizer) for both training and testing.
- **Neural Network Architectures** - Ability to use PyTorch to solve more complex problems on unstructured data such as: image classification, segmentation, object detection, text summarization, video/audio classification., etc using different neural network architectures that can be built with basic blocks available in torch.nn such as convolution layers, recurrent layers, etc.
- **Using Pre-trained models** - Ability to load and fine tune pre-trained models for different tasks.
- **Distributed Training** - Usage of multiple GPUs for training models using Model Parallelism or Data Parallelism.
- **Quantization** - Theoretical understanding and practical usage of quantization in PyTorch which increases speed and reduces memory footprint of models.
- **Debugging** - Ability to debug a model pipeline built in PyTorch and narrowing down on root cause of any failure such as parameters exploding during training, etc.
- **Deployment** - Ability to serve models as services and building REST APIs for inference.
