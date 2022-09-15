#include "NeuralNetTesting.hpp"
#include "../CPUNeuralNetwork.hpp"
#include <vector>
#include <iostream>


/*----------------------------------------------*/
// Testing methodes of the NeuralNetowrk class

/**
 * This methodes tests forward propegation of the Neural Network.
 */
void cpu::NeuralNetTesting::test_forwardPropegation(){

    cpu::Vector x = {1.0f};
    cpu::Matrix W1(2,1,{1.0f, 0.0f});
    cpu::Matrix W2(2,2,{1.0f, 0.0f, 0.0f, 0.0f});
    cpu::Vector W3 = {1.0f, 0.0f};

    cpu::NeuralNetwork net(2,2, 0, 0.01);

    net.x(x);
    net.m_hidden_layer1.W(W1);
    net.m_hidden_layer2.W(W2);
    net.m_output_layer.W(W3);

    float actual_a3 = net.forwardPropegation();

    float expected_a3 = 0.731f;


    if(areFloatEqual(expected_a3, actual_a3)){
        std::cout << "Test passed! Forward propegation produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! Forward propegation produced unexpected results." << std::endl;
    }
}

/**
 * 
 * This methode tests if the correct gradient is computed.
 * The finite difference will be used to approximate the expected gradient. 
 * 
 * The algorithm is as follows:
 * 1. Radomly initialize weight of neural network from a Guassian distribution. 
 * 2. Perform forward and backpropegation to determine the gradients computed
 *    for the last layer of the neural network and store the result. 
 * 3. for each layer:
 *          For each gradient w of the last weight:
 *              - compute negative perturbation: w_minus = w- perturb
 *              - perform forward propegation of neural network 
 *                with w_minus instead of w
 *              - compute loss_minus which is the loss of the neural network
 *                 by replacing w with w_minus.
 *              - compute positve pertubation: w_positive = w + perturb
 *              - perform forward propegation of neural network with 
 *                w_positive instead of w
 *              - compute loss_positive which is the loss of the neural network
 *                by replacing w with w_positive.
 *              - estimate numerical gradient for w by using the finite difference methode
 *                numericGradient = (loss_positive - loss_negative)/2*perturb
 *              - store numericGradient.
 * 4. Compare the gradients for the last layer produced by backpropegation
 *    with the numerical estimated gradients.
 */
void cpu::NeuralNetTesting::test_backPropegation(){

    cpu::NeuralNetwork net(10,10, 0, 0.01);

    cpu::Vector x = {-2.11764, 0.3571 , -0.423171};
    float y = 0;

    cpu::Matrix W1(10,3);
    cpu::Matrix W2(10,10);
    cpu::Vector W3(10,0.0f);


    cpu::Matrix W1_minus(10,3);
    cpu::Matrix W1_plus(10,3);

    cpu::Matrix W2_minus(10,10);
    cpu::Matrix W2_plus(10,10);

    cpu::Vector W3_minus(10, 0.0f);
    cpu::Vector W3_plus(10, 0.0f);

    cpu::Matrix numericdLdW1(10,3);
    cpu::Matrix numericdLdW2(10,10);
    cpu::Vector numericdLdW3(10, 0.0f);

    float perturb = 0.0001;

    float loss_minus;
    float loss_plus;

    W1.matrixInitialization();
    W2.matrixInitialization();
    W3.vectorInitialization();

    net.x(x);
    net.m_hidden_layer1.W(W1);
    net.m_hidden_layer2.W(W2);
    net.m_output_layer.W(W3);
    net.y(y);

    net.forwardPropegation();
    net.backPropegation();

    cpu::Vector actual_dLdW3 = net.m_output_layer.dLdW();
    cpu::Matrix actual_dLdW2 = net.m_hidden_layer2.dLdW();
    cpu::Matrix actual_dLdW1 = net.m_hidden_layer1.dLdW();

    for(int i=0; i < W3.getSize(); i++){
        W3_minus = W3;
        W3_plus = W3;
        W3_minus[i] -= perturb;
        W3_plus[i] += perturb;

        net.m_output_layer.W(W3_minus);
        net.forwardPropegation();
        loss_minus = net.m_output_layer.bceLoss(y);

        net.m_output_layer.W(W3_plus);
        net.forwardPropegation();
        loss_plus =net.m_output_layer.bceLoss(y);

        numericdLdW3[i] = (loss_plus-loss_minus)/(2*perturb);      
    }
    net.m_output_layer.W(W3);

    for (int j = 0; j < W2.get_num_rows(); j++){
        for(int i=0; i < W2.get_num_cols(); i++){
            W2_minus = W2;
            W2_plus = W2;
            W2_minus(j,i) -= perturb;
            W2_plus(j,i) += perturb;

            net.m_hidden_layer2.W(W2_minus);
            net.forwardPropegation();
            loss_minus = net.m_output_layer.bceLoss(y);

            net.m_hidden_layer2.W(W2_plus);
            net.forwardPropegation();
            loss_plus = net.m_output_layer.bceLoss(y);

            numericdLdW2(j,i) = (loss_plus-loss_minus)/(2*perturb);
        }
    }
    net.m_hidden_layer2.W(W2);

    for (int j = 0; j < W1.get_num_rows(); j++){
        for(int i=0; i < W1.get_num_cols(); i++){
            W1_minus = W1;
            W1_plus = W1;
            W1_minus(j,i) -= perturb;
            W1_plus(j,i) += perturb;

            net.m_hidden_layer1.W(W1_minus);
            net.forwardPropegation();
            loss_minus = net.m_output_layer.bceLoss(y);

            net.m_hidden_layer1.W(W1_plus);
            net.forwardPropegation();
            loss_plus =net.m_output_layer.bceLoss(y);

            numericdLdW1(j,i) = (loss_plus-loss_minus)/(2*perturb);
        }
    }
    

    if ( actual_dLdW3 == numericdLdW3)
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for last layer.\n";
    else
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for last layer.\n";
    
    if(actual_dLdW2 == numericdLdW2){
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for second layer.\n";
    } else{
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for second layer.\n";
    }

    if(actual_dLdW1 == numericdLdW1){
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for first layer.\n";
    } else{
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for first layer.\n";
    }
}

/**
 * This methode tests the gradient descent algorithm the derived layers classes.
 * Since both the hidden and output layers shared the same algorithm, only the
 * output layer gradientDescent methode will be test.
 * 
 * The gradient decent methode must produce a series of 
 * non-decreasing objectives in order for this test to pass.
 * The details of this test is as follows.
 * Let @f$L = w^2$ and then @f$dLdw = 2w$ with step size of @f$\alpha = 0.01$.
 * The initial starting position will be @f$w=100$ and the number of iterations
 * will be 5. The choices of these numbers are random. If after each iteration,
 * the loss for the new position is smaller than the loss for the old position,
 * then this test will pass.
 * 
 */
void cpu::NeuralNetTesting::test_gradientDescent(){

    cpu::OutputLayer outputlayer(1);
    float alpha = 0.01;

    bool testPass = true;

    int numIter = 5;

    cpu::Vector w(1, 100);
    outputlayer.W(w);
    float loss = computeQuadraticLoss(w);
    float prev_loss;
    cpu::Matrix dLdw = computeGradientQuadraticLoss(w);

    for(int i = 0; i < numIter; i++){
        prev_loss = loss;
        outputlayer.gradientDecent(alpha);
        loss = computeQuadraticLoss(w);
        if(loss > prev_loss){
            testPass = false;
        }
    }

    if(testPass){
        std::cout << "Test succeeded! Gradient descent produces expected results." << std::endl;
    }else{
        std::cout << "Test failed! Gradient descent produces unexpected results." << std::endl;
    }
}




/*----------------------------------------------*/
// Helper methodes.

/**
 * Determine if two float values are equal with a fixed error. 
 * Fixed point errors are not used for comparison between floating point values
 * but it will suffice for our usage. 
 */
bool cpu::NeuralNetTesting::areFloatEqual(float a, float b){
    constexpr float epsilon = 0.01; 
    return std::abs(a - b) < epsilon;
}

/**
 * Compute and return the quadratic loss as follows.
 * @f$L = w^2$
 * 
 */
float cpu::NeuralNetTesting::computeQuadraticLoss(cpu::Vector& w){
    float quadraticLoss = w[0]*w[0];

    return quadraticLoss;
}

/**
 * 
 * Compute and return the derivative of the
 * quadratic loss function.
 * 
 */
cpu::Matrix cpu::NeuralNetTesting::computeGradientQuadraticLoss(cpu::Vector& w){
    float gradientQuadracticLoss = 2*w[0];

    cpu::Matrix gradientQuadracticLoss_(1,1,{gradientQuadracticLoss});

    return gradientQuadracticLoss_;
}