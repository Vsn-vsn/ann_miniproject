# Phase1
1) Time Domain (t): We defined a tensor(a multidimensional array, which is a major data structure to hold weights, biases, output in a neural network).
    - t is a tensor of 200 points fro, t=0 to t=10, which is the time interval along which we want the NN to solve the equation.
2) Initial conditions (t_ic, x_ic, v_ic): This is the known data.
    - We are telling the network that at t=0, the displacement is 1 and the velocity $\frac{dx}{dt}$ is 0.0.
    - The network must learn to satisfy these points in training.
3) Collocation Points (t_collocation): This is a key concept in Physics-Informed Neural network.
    - These points are provided with an answer, instead during the training we enforce the physics of the problem onto these points.
    - During the training we see if the network can satisfy the equation at this given point. 
    - Hence, the network learns the underlying physics law and doesn't just memorize the data.

# Phase2
Here our problem is solved using a standard Multilayer Perceptron.
1) The Model (PINN_MLP):A standard multi-layer perceptron(MLP).
    - It takes one input time(denoted as t) and produces an output x(t), which is the displacement.
    - Here we have used 4 hidden layers and 32 neurons.
    - The activation fuctions used in the perceptrons is the `tanh` function. 
        - This is used because tanh is smooth and differentiable infinitely.
        - This is crucial because we need to compute the first and second derivatives to check if ODE is satisfied, and a non-smooth function like ReLU would fail.
    - Weigtht intiaisation is done through `xavier_uniform_` which helps the network to train effectively from the start.   

2) Physics-Informed Core (derivatives and pinn_loss): These are the core computation functions.
    - `derivatives` function uses the `torch.autograd.grad` method to compute $\frac{dx}{dt}$ and $\frac{d^2x}{dt^2}$ (first and second derivatives).
    - `pinn_loss` function is not just a normal loss function for the network's error between predicted and true data, but is a combination of two things:
        - Initial condition loss, which is the standard meas square error(mse). It measures how far is the network's preditcition from the true initial conditions at t=0. This forces the solution to start from the right place.
        - Physics loss, which is calculated at the collocation points. W take the network's output (x<sub>col</sub>) and calculates its derivatives(dx<sub>col</sub>, ddx<sub>col</sub>) automatically with the help of `derivatives` function. The derivatives are plugged into the ODE's formula to get the residual. ODE says residual must be zero, and hence physics loss is the MSE of this residual.     By minimizing this loss, you are forcing the network to learn a function that obeys the laws of physics.
3) The Training Loop: We use the Adam optimizer.
    - In each epoch, the optimizer's goal is to adjust the MLP's weights and biases to minimize the `total_loss` (the sum of initial condition and physics loss).
    - The process `total_loss.backward()` is the Back-propagation algorithm. It calculates how much each weight and bias in the network contributed to the total error and `optimizer.step()` updates them accordingly. 