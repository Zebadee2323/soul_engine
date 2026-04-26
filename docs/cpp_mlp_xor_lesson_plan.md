# Building a Basic C++ MLP from Scratch: XOR Lesson Plan

## Goal

Build a small configurable multi-layer perceptron in C++ that can learn the XOR function without using any machine learning libraries.

Math libraries such as GLM are allowed, but the neural-network logic should be written manually.

By the end, you should have:

- A small C++ MLP implementation
- Configurable dense layers
- Forward propagation
- ReLU and sigmoid activations
- Mean squared error or binary cross-entropy loss
- Backpropagation
- Gradient descent training
- A working XOR demo
- Basic save/load support for trained weights

The target problem is XOR:

```text
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
```

A minimal network that can learn this is:

```text
2 inputs
-> 2 to 4 hidden neurons
-> 1 output neuron
```

Example:

```text
2 -> 4 -> 1
```

---

# Phase 0: Project Setup

## Theory

Before touching neural-network math, set up a small C++ project that is easy to iterate on. Treat this like building a tiny engine subsystem: first make the core loop simple, testable, and visible.

For this project, the main executable should:

1. Create an XOR dataset
2. Create an MLP
3. Train it for many iterations
4. Print predictions before and after training

You do not need a complex build system, but using CMake is a good idea.

## C++ Code Required

Suggested initial structure:

```text
mlp-xor/
  CMakeLists.txt
  src/
    main.cpp
    math.hpp
    dense_layer.hpp
    dense_layer.cpp
    mlp.hpp
    mlp.cpp
    activations.hpp
    loss.hpp
    dataset.hpp
```

At first, many of these files can be empty placeholders.

If using GLM, you can represent vectors and matrices with:

```cpp
std::vector<float>
glm::vec<N>
glm::mat<C, R, float>
```

However, because we want configurable layer sizes, dynamic containers are simpler than fixed-size GLM types.

Recommended for the first version:

```cpp
using Vector = std::vector<float>;
using Matrix = std::vector<std::vector<float>>;
```

This is not the fastest layout, but it is easy to debug. Later you can replace it with a flat array.

## Tasks

- [X] Create a new C++ project folder.
- [X] Add a `CMakeLists.txt` or equivalent build setup.
- [X] Create `main.cpp`.
- [X] Add placeholder headers for the neural-network components.
- [X] Make the project compile and print `Hello MLP`.
- [X] Decide whether to use plain `std::vector<float>` or GLM for early math. 
        (Decision: we'll just use plain std::vector<float> for now)
- [ ] Create a simple `Vector` alias.
- [ ] Create a simple `Matrix` alias.

---

# Phase 1: Representing Vectors, Matrices, and Layer Shapes

## Theory

An MLP is mostly repeated matrix-vector multiplication.

A dense layer computes:

```text
y = W x + b
```

Where:

```text
x = input vector
W = weight matrix
b = bias vector
y = output vector
```

If a layer has 2 inputs and 4 outputs:

```text
input size  = 2
output size = 4
```

Then:

```text
x shape = 2
W shape = 4 x 2
b shape = 4
y shape = 4
```

In game-dev terms, this is like transforming a vector through a matrix, except the matrix is not necessarily 4x4 and the meaning is not spatial. Instead of transforming position, rotation, or scale, you are transforming feature values into neuron activations.

Each output neuron owns one row of the matrix:

```text
y[0] = W[0][0] * x[0] + W[0][1] * x[1] + b[0]
y[1] = W[1][0] * x[0] + W[1][1] * x[1] + b[1]
y[2] = W[2][0] * x[0] + W[2][1] * x[1] + b[2]
y[3] = W[3][0] * x[0] + W[3][1] * x[1] + b[3]
```

## C++ Code Required

Create basic utility functions:

```cpp
using Vector = std::vector<float>;
using Matrix = std::vector<std::vector<float>>;

Vector make_vector(size_t size, float value = 0.0f);
Matrix make_matrix(size_t rows, size_t cols, float value = 0.0f);

Vector mat_vec_mul(const Matrix& m, const Vector& v);
Vector add(const Vector& a, const Vector& b);
```

You also want simple shape checking during development:

```cpp
assert(m[0].size() == v.size());
assert(a.size() == b.size());
```

## Tasks

- [ ] Define `using Vector = std::vector<float>;`.
- [ ] Define `using Matrix = std::vector<std::vector<float>>;`.
- [ ] Implement `make_vector(size, value)`.
- [ ] Implement `make_matrix(rows, cols, value)`.
- [ ] Implement `mat_vec_mul(matrix, vector)`.
- [ ] Implement vector addition.
- [ ] Add assert-based shape checks.
- [ ] Write a small test in `main.cpp` that multiplies a `2x2` matrix by a 2D vector.
- [ ] Print the result and verify it manually.

---

# Phase 2: Dense Layer Forward Pass

## Theory

A dense layer is a group of neurons where every input connects to every output.

For a layer:

```text
input size:  2
output size: 4
```

The layer stores:

```text
weights: 4 x 2
biases:  4
```

The forward pass is:

```text
z = W x + b
```

The value `z` is often called the pre-activation value. Later we will apply an activation function to get:

```text
a = activation(z)
```

For now, implement only the linear part.

Think of this as the neural-network version of a transform component. The layer owns parameters, and `forward()` applies those parameters to an input vector.

## C++ Code Required

Create a `DenseLayer` class:

```cpp
class DenseLayer
{
public:
    DenseLayer(size_t input_size, size_t output_size);

    Vector forward(const Vector& input);

private:
    size_t m_input_size;
    size_t m_output_size;
    Matrix m_weights;
    Vector m_biases;
};
```

For now, initialize weights and biases manually or to small hardcoded values.

## Tasks

- [ ] Create `dense_layer.hpp`.
- [ ] Create `dense_layer.cpp`.
- [ ] Add `input_size` and `output_size` members.
- [ ] Add `weights` and `biases` members.
- [ ] Implement a constructor that creates the correct weight and bias shapes.
- [ ] Implement `DenseLayer::forward()`.
- [ ] In `main.cpp`, create a `DenseLayer` with `2` inputs and `3` outputs.
- [ ] Feed it an input like `{1.0f, 2.0f}`.
- [ ] Print the output vector.
- [ ] Verify that output size matches the layer output size.

---

# Phase 3: Random Weight Initialization

## Theory

Neural networks need random initial weights.

If every weight starts at zero, all neurons in a layer behave identically. They receive the same gradients and learn the same thing. This is called symmetry, and it prevents hidden layers from becoming useful.

So we initialize weights with small random values.

For a beginner implementation, this is fine:

```text
weights in range [-1, 1]
biases = 0
```

Later, you can improve this with Xavier or He initialization.

A decent simple initialization for sigmoid/tanh is Xavier-style:

```text
limit = sqrt(6 / (input_size + output_size))
weight = random(-limit, limit)
```

For ReLU, He initialization is common:

```text
stddev = sqrt(2 / input_size)
```

For XOR, simple random values are enough.

## C++ Code Required

Use the standard random library:

```cpp
#include <random>

std::mt19937 rng(seed);
std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
```

Add a helper:

```cpp
float random_float(float min, float max);
```

Then initialize weights in the `DenseLayer` constructor.

## Tasks

- [ ] Add a random-number helper.
- [ ] Choose a fixed seed for reproducible debugging.
- [ ] Initialize weights randomly.
- [ ] Initialize biases to `0.0f`.
- [ ] Print the initial weights of a small layer.
- [ ] Run the program multiple times with the same seed and confirm the same values are generated.
- [ ] Change the seed and confirm different values are generated.

---

# Phase 4: Activation Functions

## Theory

Without activation functions, stacking layers does not give you a real deep network.

This is because multiple linear transforms collapse into one linear transform:

```text
W2(W1x + b1) + b2
```

is still just another linear function.

XOR is not linearly separable, so a purely linear model cannot learn it.

Activation functions introduce nonlinearity.

For XOR, a classic choice is sigmoid:

```text
sigmoid(x) = 1 / (1 + e^-x)
```

Sigmoid maps any number into the range:

```text
0 to 1
```

This makes it convenient for binary output.

For hidden layers, ReLU is also common:

```text
ReLU(x) = max(0, x)
```

For your first XOR network, using sigmoid for both hidden and output layers keeps the backpropagation easier to reason about.

## C++ Code Required

Create an activation enum:

```cpp
enum class ActivationType
{
    None,
    Sigmoid,
    ReLU
};
```

Implement:

```cpp
float activate(float x, ActivationType type);
float activation_derivative_from_output(float activated_value, ActivationType type);
```

For sigmoid, if `y = sigmoid(x)`, then:

```text
sigmoid'(x) = y * (1 - y)
```

That is convenient because you can compute the derivative from the already-activated output.

For ReLU:

```text
ReLU'(x) = 1 if x > 0 else 0
```

For ReLU, it is usually better to use the pre-activation value `z`, not the activated output.

To keep the first implementation simple, start with sigmoid only.

## Tasks

- [ ] Create `activations.hpp`.
- [ ] Add `ActivationType` enum.
- [ ] Implement `sigmoid(x)`.
- [ ] Implement `sigmoid_derivative_from_output(y)`.
- [ ] Optionally implement `relu(x)`.
- [ ] Optionally implement `relu_derivative_from_pre_activation(x)`.
- [ ] Add an `ActivationType` member to `DenseLayer`.
- [ ] Update `DenseLayer::forward()` to apply activation after `W x + b`.
- [ ] Test sigmoid with known values.
- [ ] Confirm `sigmoid(0)` returns approximately `0.5`.

---

# Phase 5: Building the MLP Container

## Theory

An MLP is a sequence of layers.

Instead of manually calling:

```cpp
h = layer1.forward(input);
y = layer2.forward(h);
```

Create a `Mlp` class that owns layers and forwards through them.

This is similar to a small game pipeline:

```text
input data
-> system A
-> system B
-> system C
-> result
```

The MLP should not care whether it has one hidden layer or many. It just loops over layers.

## C++ Code Required

Create a layer config:

```cpp
struct LayerConfig
{
    size_t input_size;
    size_t output_size;
    ActivationType activation;
};
```

Create an MLP:

```cpp
class Mlp
{
public:
    Mlp(const std::vector<LayerConfig>& configs);

    Vector forward(const Vector& input);

private:
    std::vector<DenseLayer> m_layers;
};
```

Example construction:

```cpp
Mlp mlp({
    {2, 4, ActivationType::Sigmoid},
    {4, 1, ActivationType::Sigmoid}
});
```

## Tasks

- [ ] Create `mlp.hpp`.
- [ ] Create `mlp.cpp`.
- [ ] Create `LayerConfig`.
- [ ] Add `std::vector<DenseLayer>` to `Mlp`.
- [ ] Implement the `Mlp` constructor from layer configs.
- [ ] Implement `Mlp::forward()`.
- [ ] Create a `2 -> 4 -> 1` MLP in `main.cpp`.
- [ ] Feed it all four XOR inputs.
- [ ] Print the raw predictions.
- [ ] Confirm the program runs even though predictions are initially wrong.

---

# Phase 6: Dataset and Loss Function

## Theory

Training requires a way to measure error.

For XOR, each sample has:

```text
input:  [x0, x1]
target: [y]
```

Example:

```text
input  = [0, 1]
target = [1]
```

A simple first loss is mean squared error:

```text
MSE = (prediction - target)^2
```

For a vector output:

```text
MSE = average((prediction[i] - target[i])^2)
```

For a single XOR output, it is just:

```text
(prediction - target)^2
```

The derivative of MSE with respect to prediction is:

```text
dLoss/dPrediction = 2 * (prediction - target)
```

You need this derivative because backpropagation starts at the output and moves backward through the network.

## C++ Code Required

Create:

```cpp
struct Sample
{
    Vector input;
    Vector target;
};

using Dataset = std::vector<Sample>;
```

Create loss helpers:

```cpp
float mean_squared_error(const Vector& prediction, const Vector& target);
Vector mean_squared_error_derivative(const Vector& prediction, const Vector& target);
```

Create the XOR dataset:

```cpp
Dataset xor_data = {
    {{0.0f, 0.0f}, {0.0f}},
    {{0.0f, 1.0f}, {1.0f}},
    {{1.0f, 0.0f}, {1.0f}},
    {{1.0f, 1.0f}, {0.0f}},
};
```

## Tasks

- [ ] Create `dataset.hpp`.
- [ ] Define `Sample`.
- [ ] Define `Dataset`.
- [ ] Create the XOR dataset.
- [ ] Create `loss.hpp`.
- [ ] Implement `mean_squared_error()`.
- [ ] Implement `mean_squared_error_derivative()`.
- [ ] Run the Mlp over the XOR dataset.
- [ ] Print prediction, target, and loss for each sample.
- [ ] Print average loss across the dataset.

---

# Phase 7: Backpropagation for One Dense Layer

## Theory

Backpropagation is the process of asking:

```text
How much did each parameter contribute to the error?
```

A dense layer computes:

```text
z = W x + b
a = activation(z)
```

During training, you need gradients for:

```text
dW = how loss changes with each weight
db = how loss changes with each bias
dx = how loss changes with each input
```

The backward pass receives:

```text
dL/da
```

That means:

```text
how much the loss changes with respect to this layer's output activation
```

Then it computes:

```text
dL/dz = dL/da * activation_derivative(z)
```

For each weight:

```text
dL/dW[i][j] = dL/dz[i] * input[j]
```

For each bias:

```text
dL/db[i] = dL/dz[i]
```

For each input:

```text
dL/dx[j] = sum over i of W[i][j] * dL/dz[i]
```

This `dL/dx` is passed to the previous layer.

In game-dev terms, forward propagation is evaluating a graph from input to output. Backpropagation is walking the graph backward and computing how sensitive the final error is to each intermediate value.

## C++ Code Required

The layer must cache values from the forward pass:

```cpp
Vector m_last_input;
Vector m_last_z;
Vector m_last_activation;
```

Add gradient storage:

```cpp
Matrix m_weight_gradients;
Vector m_bias_gradients;
```

Add:

```cpp
Vector DenseLayer::backward(const Vector& output_gradient);
```

Where `output_gradient` means `dL/da` for this layer.

Inside `backward()`:

1. Compute `d_z`
2. Compute `weight_gradients`
3. Compute `bias_gradients`
4. Compute `input_gradient`
5. Return `input_gradient`

## Tasks

- [ ] Store `last_input` during `forward()`.
- [ ] Store `last_z` during `forward()`.
- [ ] Store `last_activation` during `forward()`.
- [ ] Add `weight_gradients` to `DenseLayer`.
- [ ] Add `bias_gradients` to `DenseLayer`.
- [ ] Implement `DenseLayer::backward(output_gradient)`.
- [ ] Compute `d_z` using activation derivative.
- [ ] Compute gradients for all weights.
- [ ] Compute gradients for all biases.
- [ ] Compute and return `input_gradient`.
- [ ] Test backward pass on a single-layer network.
- [ ] Print gradient values to confirm they are not all zero.

---

# Phase 8: Updating Weights with Gradient Descent

## Theory

Once you have gradients, training is simple:

```text
parameter = parameter - learning_rate * gradient
```

For weights:

```text
W[i][j] = W[i][j] - learning_rate * dW[i][j]
```

For biases:

```text
b[i] = b[i] - learning_rate * db[i]
```

The learning rate controls step size.

Too small:

```text
training is very slow
```

Too large:

```text
training may explode or bounce around
```

For XOR, try values like:

```text
0.1
0.5
1.0
```

If training fails, reduce the learning rate.

## C++ Code Required

Add to `DenseLayer`:

```cpp
void apply_gradients(float learning_rate);
```

Then update every weight and bias.

Eventually, you may want gradients accumulated across a batch. For the first version, update after each sample.

This is stochastic gradient descent.

## Tasks

- [ ] Add `DenseLayer::apply_gradients(float learning_rate)`.
- [ ] Update every weight using its gradient.
- [ ] Update every bias using its gradient.
- [ ] Add `Mlp::backward(loss_gradient)` that loops through layers in reverse.
- [ ] Add `Mlp::apply_gradients(learning_rate)`.
- [ ] Run one training step on a single XOR sample.
- [ ] Print loss before and after one update.
- [ ] Confirm the loss changes.

---

# Phase 9: Training the Full XOR Network

## Theory

Training means repeatedly showing the network examples and adjusting weights.

The high-level loop is:

```text
for epoch in epochs:
    total_loss = 0

    for sample in dataset:
        prediction = mlp.forward(sample.input)
        loss = compute_loss(prediction, sample.target)
        gradient = compute_loss_derivative(prediction, sample.target)
        mlp.backward(gradient)
        mlp.apply_gradients(learning_rate)

        total_loss += loss

    print average loss sometimes
```

For XOR, you may need thousands of epochs.

A typical result after training:

```text
[0, 0] -> 0.02
[0, 1] -> 0.97
[1, 0] -> 0.98
[1, 1] -> 0.03
```

Because the output uses sigmoid, interpret values near `0` as false and values near `1` as true.

## C++ Code Required

Create a training function:

```cpp
void train(
    Mlp& mlp,
    const Dataset& dataset,
    size_t epochs,
    float learning_rate);
```

Inside it:

- Run forward
- Compute loss
- Compute loss gradient
- Run backward
- Apply gradients
- Print average loss every N epochs

## Tasks

- [ ] Add a `train()` function.
- [ ] Train for `1000` epochs.
- [ ] Print average loss every `100` epochs.
- [ ] Increase to `5000` or `10000` epochs if needed.
- [ ] Try learning rate `0.1`.
- [ ] Try learning rate `0.5`.
- [ ] Try hidden sizes `2`, `3`, and `4`.
- [ ] Print predictions after training.
- [ ] Confirm XOR predictions are close to correct.
- [ ] Add a threshold such as `prediction > 0.5` to convert output to class result.

---

# Phase 10: Debugging and Numerical Sanity Checks

## Theory

Backpropagation bugs are common. The best way to debug them is to make the system observable.

Common failure cases:

```text
loss never changes
loss becomes NaN
all predictions become the same
weights explode to huge numbers
network gets stuck around 0.5
```

Useful checks:

- Are gradients non-zero?
- Are weights changing?
- Are inputs and targets correct?
- Is the activation derivative correct?
- Is the learning rate too high?
- Are matrix shapes correct?

For sigmoid, a common issue is saturation. If inputs to sigmoid become very large positive or negative numbers, the derivative becomes tiny, and learning slows down.

## C++ Code Required

Add debug helpers:

```cpp
void print_vector(const Vector& v);
void print_matrix(const Matrix& m);
float vector_min(const Vector& v);
float vector_max(const Vector& v);
bool contains_nan(const Vector& v);
```

Optional:

```cpp
void Mlp::print_weights() const;
void Mlp::print_gradient_stats() const;
```

## Tasks

- [ ] Add `print_vector()`.
- [ ] Add `print_matrix()`.
- [ ] Add NaN checks for predictions.
- [ ] Add NaN checks for loss.
- [ ] Print average loss every N epochs.
- [ ] Print predictions before and after training.
- [ ] Print a warning if loss becomes NaN.
- [ ] Print weight ranges occasionally.
- [ ] Test with a very small learning rate.
- [ ] Test with a too-large learning rate and observe failure behavior.

---

# Phase 11: Refactoring Toward a Cleaner MLP API

## Theory

Once the XOR system works, clean the design.

The goal is to move from a demo into a reusable component.

A useful API might look like:

```cpp
Mlp mlp({
    {2, 4, ActivationType::Sigmoid},
    {4, 1, ActivationType::Sigmoid}
});

mlp.train(dataset, TrainConfig{
    .epochs = 10000,
    .learning_rate = 0.5f
});

Vector output = mlp.predict({0.0f, 1.0f});
```

Avoid over-engineering too early. The first goal is correctness. After that, improve ergonomics.

## C++ Code Required

Create:

```cpp
struct TrainConfig
{
    size_t epochs = 10000;
    float learning_rate = 0.5f;
    size_t log_every = 1000;
};
```

Add:

```cpp
Vector Mlp::predict(const Vector& input);
void Mlp::train(const Dataset& dataset, const TrainConfig& config);
```

## Tasks

- [ ] Add `TrainConfig`.
- [ ] Move training logic into `Mlp::train()` or a dedicated trainer.
- [ ] Add `Mlp::predict()`.
- [ ] Make layer configuration clean and readable.
- [ ] Remove temporary debug prints or guard them behind a flag.
- [ ] Add comments explaining shape conventions.
- [ ] Add assertions for invalid architecture configs.
- [ ] Make sure the project still learns XOR after refactoring.

---

# Phase 12: Save and Load Weights

## Theory

Training produces useful parameters:

```text
weights
biases
```

To use the model later, you need to save those values.

A saved model should include:

```text
layer count
input/output size per layer
activation type per layer
weights per layer
biases per layer
```

For a first implementation, a plain text format is fine. This makes it easy to inspect and debug.

Later, you can use binary serialization.

## C++ Code Required

Add:

```cpp
void Mlp::save(const std::string& path) const;
static Mlp Mlp::load(const std::string& path);
```

Simple text format example:

```text
layers 2
layer 2 4 sigmoid
weights
...
biases
...
layer 4 1 sigmoid
weights
...
biases
...
```

## Tasks

- [ ] Add save function to `DenseLayer` or `Mlp`.
- [ ] Add load function to `DenseLayer` or `Mlp`.
- [ ] Save architecture information.
- [ ] Save activation type.
- [ ] Save weights.
- [ ] Save biases.
- [ ] Train XOR and save the model.
- [ ] Restart the program and load the model.
- [ ] Confirm loaded model produces the same predictions.

---

# Phase 13: Optional Performance-Oriented Refactor

## Theory

The initial version prioritizes clarity. Once it works, you can make the data layout more cache-friendly.

Instead of:

```cpp
std::vector<std::vector<float>> weights;
```

Use a flat vector:

```cpp
std::vector<float> weights;
```

With indexing:

```cpp
weights[row * input_size + col]
```

This improves locality and avoids many small allocations.

In game-engine terms, this is the same reason you might prefer a contiguous component array over many separately allocated objects.

## C++ Code Required

Change:

```cpp
Matrix m_weights;
Matrix m_weight_gradients;
```

Into:

```cpp
std::vector<float> m_weights;
std::vector<float> m_weight_gradients;
```

Add helper:

```cpp
size_t weight_index(size_t output_neuron, size_t input_neuron) const
{
    return output_neuron * m_input_size + input_neuron;
}
```

## Tasks

- [ ] Replace nested matrix storage with flat vector storage.
- [ ] Add `weight_index(row, col)` helper.
- [ ] Update forward pass.
- [ ] Update backward pass.
- [ ] Update gradient application.
- [ ] Update save/load.
- [ ] Confirm XOR still trains correctly.
- [ ] Benchmark before and after if desired.

---

# Phase 14: Optional Mini-Batch Training

## Theory

The first training loop updates weights after every sample. This is stochastic gradient descent.

Mini-batch training instead accumulates gradients over several samples, then applies one averaged update.

For tiny XOR, this does not matter much. But for real datasets, mini-batches are standard.

The idea:

```text
clear accumulated gradients
for sample in batch:
    forward
    backward
    accumulate gradients
average gradients
apply update
```

## C++ Code Required

You need separate storage for accumulated gradients or a way to add into existing gradient buffers.

Add:

```cpp
void DenseLayer::zero_gradients();
void DenseLayer::accumulate_gradients(...);
void DenseLayer::apply_gradients(float learning_rate, float scale);
```

Where `scale` might be:

```text
1.0 / batch_size
```

## Tasks

- [ ] Add `zero_gradients()`.
- [ ] Change backward pass to accumulate instead of overwrite, or add a mode for accumulation.
- [ ] Add batch size to `TrainConfig`.
- [ ] Average gradients over the batch.
- [ ] Apply gradients once per batch.
- [ ] Confirm XOR still trains.
- [ ] Compare stochastic training vs full-batch training on XOR.

---

# Phase 15: Preparing for Audio Features Later

## Theory

Once XOR works, the MLP does not care where inputs come from.

The eventual audio-emotion version might use:

```text
RMS
spectral centroid
zero-crossing rate
pitch stability
```

That means input size:

```text
4
```

If you classify four emotions:

```text
happy
sad
angry
neutral
```

Then output size could be:

```text
4
```

For multi-class classification, you would usually use:

```text
softmax output
cross-entropy loss
```

But for XOR, sigmoid plus MSE is simpler.

The biggest practical issue for audio features will be normalization. Features should be scaled into comparable ranges, often around:

```text
0 to 1
```

or standardized as:

```text
mean 0, standard deviation 1
```

## C++ Code Required

Eventually add:

```cpp
struct NormalizationStats
{
    Vector mean;
    Vector stddev;
};

Vector normalize(const Vector& input, const NormalizationStats& stats);
```

For classification labels, use one-hot vectors:

```cpp
happy   = {1, 0, 0, 0};
sad     = {0, 1, 0, 0};
angry   = {0, 0, 1, 0};
neutral = {0, 0, 0, 1};
```

## Tasks

- [ ] Keep the MLP independent from XOR-specific code.
- [ ] Make input size configurable.
- [ ] Make output size configurable.
- [ ] Add a placeholder for normalization stats.
- [ ] Add a placeholder for class labels.
- [ ] Research softmax after XOR is complete.
- [ ] Research cross-entropy loss after XOR is complete.
- [ ] Try a fake 4-input dataset before using real audio features.

---

# Suggested Implementation Order Summary

Use this as the main checklist.

- [ ] Project compiles.
- [ ] Basic vector and matrix helpers work.
- [ ] Dense layer forward pass works.
- [ ] Random initialization works.
- [ ] Sigmoid activation works.
- [ ] MLP can contain multiple layers.
- [ ] XOR dataset exists.
- [ ] MSE loss works.
- [ ] MSE derivative works.
- [ ] Dense layer backward pass works.
- [ ] Gradient descent updates weights.
- [ ] Full MLP trains on XOR.
- [ ] Predictions become correct after training.
- [ ] Model can be saved.
- [ ] Model can be loaded.
- [ ] Code is refactored into a clean API.

---

# Recommended First Milestone

The first real success condition is:

```text
Before training:
[0, 0] -> around random / 0.5
[0, 1] -> around random / 0.5
[1, 0] -> around random / 0.5
[1, 1] -> around random / 0.5
```

After training:

```text
[0, 0] -> close to 0
[0, 1] -> close to 1
[1, 0] -> close to 1
[1, 1] -> close to 0
```

Good enough threshold:

```text
prediction < 0.2 means 0
prediction > 0.8 means 1
```

Excellent threshold:

```text
prediction < 0.05 means 0
prediction > 0.95 means 1
```

---

# Minimal Architecture to Aim For

Start with:

```cpp
Mlp mlp({
    {2, 4, ActivationType::Sigmoid},
    {4, 1, ActivationType::Sigmoid}
});
```

Training config:

```cpp
TrainConfig config;
config.epochs = 10000;
config.learning_rate = 0.5f;
config.log_every = 1000;
```

Expected final behavior:

```text
Input: [0, 0], prediction: 0.02
Input: [0, 1], prediction: 0.98
Input: [1, 0], prediction: 0.97
Input: [1, 1], prediction: 0.03
```

Exact values will differ depending on random seed and learning rate.

---

# Concepts You Should Understand by the End

- [ ] What a dense layer is.
- [ ] What weights and biases are.
- [ ] What a forward pass is.
- [ ] Why activation functions are needed.
- [ ] Why XOR requires a hidden layer.
- [ ] What a loss function is.
- [ ] What a gradient is.
- [ ] What backpropagation does.
- [ ] How gradient descent updates parameters.
- [ ] Why learning rate matters.
- [ ] Why initialization matters.
- [ ] How to save and reload trained parameters.
- [ ] How this can later generalize to audio-emotion classification.

---

# Notes for the Later Audio Version

When you move from XOR to audio emotion classification, the MLP structure might look like:

```text
4 input features
-> 16 hidden neurons
-> 16 hidden neurons
-> 4 emotion outputs
```

Example:

```cpp
Mlp emotion_model({
    {4, 16, ActivationType::ReLU},
    {16, 16, ActivationType::ReLU},
    {16, 4, ActivationType::Softmax}
});
```

But do not start there.

Start with XOR because it proves:

```text
forward pass works
activation works
backpropagation works
training works
```

Once XOR works, the remaining work is mostly data preparation, better loss functions, and better evaluation.
