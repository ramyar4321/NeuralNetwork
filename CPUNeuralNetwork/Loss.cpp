#include "Loss.hpp"
#include <cmath>

/**
 * Constructor for Loss class.
 * Since Loss class has no member variables, 
 * the constructor is empty.
 */
cpu::Loss::Loss(){}

/**
 * 
 * Compute the loss of the neural network using the 
 * Cross-Entropy loss function. An epsilon value is added in
 * the case that the activation is zero.
 * 
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 * 
 * @param a The vector containing the activation of the output layer
 * @param y The actual outcome from the dataset
 * 
 * @return The entropy loss 
 * 
 */
double cpu::Loss::computeLoss(const cpu::Vector& a, const cpu::Vector& y){
    double loss = 0.0f;
    // Use epsilon since log of zero is undefined.
    double epsilon = 0.0001; 

    for(int j = 0; j < a.getSize(); j ++){
        loss += -y[j]*std::log(a[j] + epsilon) - (1-y[j])*std::log(1-a[j] + epsilon);
    }

    return loss;
}

/**
 * Compute the derivative of binary cross entropy loss function
 * @f$\frac{\partial L}{\partial a} = - \frac{y}{a} + \fra{1-y}{1-a}$ where
 * @f$a$ is the output of the output neuron.
 * Since division by zero can cause numerical issues, a small value, called epsilon,
 * will be added to the denominator terms.
 * 
 * @param a The vector containing the activation of the output layer
 * @param y The actual outcome from the dataset
 * 
 * @return The derivative of the cross entropy loss function with
 *         respect to the sigmoid activation
 */
double cpu::Loss::computeLossPrime(const cpu::Vector& a, const cpu::Vector& y){

    double loss = 0.0;
    double epsilon = 0.0001;

    for(int j=0; j < a.getSize(); j++){
        loss += -(y[j]/(a[j]+epsilon)) + ((1-y[j])/(1-a[j]+epsilon));
    }

    return loss;
}