#include "NeuralNetTesting.hpp"
#include "../NeuralNetwork.hpp"
#include <vector>
#include <iostream>


/*----------------------------------------------*/
// Testing methodes of the NeuralNetowrk class

/**
 * This methodes tests forward propegation of the Neural Network.
 */
void gpu::NeuralNetTesting::test_forwardPropegation(){

    std::vector<float> x_ = {1.0f};
    std::vector<float> W1_ = {1.0f, 0.0f};
    std::vector<float> W2_ = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> W3_ = {1.0f, 0.0f};

    gpu::Vector x(x_);
    gpu::Matrix W1(2,1,W1_);
    gpu::Matrix W2(2,2,W2_);
    gpu::Vector W3(W3_);

    gpu::NeuralNetwork net(2,2, 0, 0.01);


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
void gpu::NeuralNetTesting::test_backPropegation(){

    gpu::NeuralNetwork net(10,10, 0, 0.01);

    std::vector<float> x_ = {2.11764f, 0.3571f , -0.423171f};
    gpu::Vector x(x_);
    float y = 0;

    gpu::Matrix W1(10,3);
    gpu::Matrix W2(10,10);
    gpu::Vector W3(10);


    gpu::Matrix W1_minus(10,3);
    gpu::Matrix W1_plus(10,3);

    gpu::Matrix W2_minus(10,10);
    gpu::Matrix W2_plus(10,10);

    gpu::Vector W3_minus(10);
    gpu::Vector W3_plus(10);

    gpu::Matrix numericdLdW1(10,3);
    gpu::Matrix numericdLdW2(10,10);
    gpu::Vector numericdLdW3(10);

    float perturb = 0.0001;

    float loss_minus;
    float loss_plus;


    W1.matrixInitializationDevice();
    W2.matrixInitializationDevice();
    W3.vectorInitializationDevice();

    net.x(x);
    net.m_hidden_layer1.W(W1);
    net.m_hidden_layer2.W(W2);
    net.m_output_layer.W(W3);
    net.y(y);

    net.forwardPropegation();
    net.backPropegation();


    gpu::Vector actual_dLdW3 = net.m_output_layer.dLdW();
    gpu::Matrix actual_dLdW2 = net.m_hidden_layer2.dLdW();
    gpu::Matrix actual_dLdW1 = net.m_hidden_layer1.dLdW();

    for(int i=0; i < W3.getSize(); i++){
        W3_minus.deepCopy(W3);
        W3_plus.deepCopy(W3);
        W3_minus[i] -= perturb;
        W3_plus[i] += perturb;

        net.m_output_layer.WDeepCopy(W3_minus);
        net.forwardPropegation();
        loss_minus = net.m_output_layer.bceLoss(y);

        net.m_output_layer.WDeepCopy(W3_plus);
        net.forwardPropegation();
        loss_plus =net.m_output_layer.bceLoss(y);

        numericdLdW3[i] = (loss_plus-loss_minus)/(2*perturb);      
    }
    net.m_output_layer.W(W3);

    for (int j = 0; j < W2.get_num_rows(); j++){
        for(int i=0; i < W2.get_num_cols(); i++){
            W2_minus.deepCopy(W2);
            W2_plus.deepCopy(W2);
            W2_minus(j,i) -= perturb;
            W2_plus(j,i) += perturb;

            net.m_hidden_layer2.WDeepCopy(W2_minus);
            net.forwardPropegation();
            loss_minus = net.m_output_layer.bceLoss(y);

            net.m_hidden_layer2.WDeepCopy(W2_plus);
            net.forwardPropegation();
            loss_plus = net.m_output_layer.bceLoss(y);

            numericdLdW2(j,i) = (loss_plus-loss_minus)/(2*perturb);
        }
    }
    net.m_hidden_layer2.W(W2);

    for (int j = 0; j < W1.get_num_rows(); j++){
        for(int i=0; i < W1.get_num_cols(); i++){
            W1_minus.deepCopy(W1);
            W1_plus.deepCopy(W1);
            W1_minus(j,i) -= perturb;
            W1_plus(j,i) += perturb;

            net.m_hidden_layer1.WDeepCopy(W1_minus);
            net.forwardPropegation();
            loss_minus = net.m_output_layer.bceLoss(y);

            net.m_hidden_layer1.WDeepCopy(W1_plus);
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
void gpu::NeuralNetTesting::test_gradientDescent(){

    gpu::OutputLayer outputlayer(1);
    float alpha = 0.01;

    bool testPass = true;

    int numIter = 5;

    std::vector<float> w_ = {100.0f};
    gpu::Vector w(w_);
    outputlayer.W(w);
    float loss = computeQuadraticLoss(w);
    float prev_loss;
    gpu::Matrix dLdw = computeGradientQuadraticLoss(w);

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
bool gpu::NeuralNetTesting::areFloatEqual(float a, float b){
    constexpr float epsilon = 0.01; 
    return std::abs(a - b) < epsilon;
}

/**
 * Compute and return the quadratic loss as follows.
 * @f$L = w^2$
 * 
 */
float gpu::NeuralNetTesting::computeQuadraticLoss(gpu::Vector& w){
    float quadratic_loss = w[0]*w[0];

    return quadratic_loss;
}

/**
 * 
 * Compute and return the derivative of the
 * quadratic loss function.
 * 
 */
gpu::Matrix gpu::NeuralNetTesting::computeGradientQuadraticLoss(gpu::Vector& w){
    std::vector<float> gradient_quadracticLoss = {2*w[0]};


    gpu::Matrix gradient_quadracticLoss_(1,1,gradient_quadracticLoss);

    return gradient_quadracticLoss_;
}
