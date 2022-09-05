#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

/**
 * Initialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(int hidden_layer1_size,
                                  int hidden_layer2_size,
                                  int epoch,
                                  double alpha):
                                  m_hidden_layer1(3, hidden_layer1_size),
                                  m_hidden_layer2(hidden_layer1_size, hidden_layer2_size),
                                  m_output_layer(hidden_layer2_size),
                                  m_epoch(epoch),
                                  m_alpha(alpha)
{}

/**
 * Use data to train the neural network.
 * 
 * @param X_train_stand The X train dataset used to train the Neural Network.
 *                      The X train dataset is assumed to be z-score standardized.
 * @param y_train       The y train dataset used to train the Neural Network.
 *                      The y train dataset is assumed to have values of 0 or 1.
 */
void cpu::NeuralNetwork::fit(cpu::Dataset& X_train_stand, std::vector<double>& y_train){

    // Initialize the weigths of neural network
    this->m_hidden_layer1.weightInitialization();
    this->m_hidden_layer2.weightInitialization();
    this->m_output_layer.weightInitialization();
    
    // Use variable to store the sample from the dataset 
    // to be passed to forward propegation methode.
    // Each sample is a vector of length three since the 
    // Haberman dataset has three features. 
    cpu::Vector x(3, 0.0);

    // Use variable to store the outcome associated
    // with each given sample from the dataset to be 
    // passed to backward propegation.
    double y;

    for (int e =0; e< this->m_epoch; e++){
        for (int j=0; j < X_train_stand.get_num_rows(); j++){
            x = X_train_stand.getRow(j);
            y = y_train[j];

            this->forward_propegation(x);
            this->backPropegation(x, y);
            this->updateWeigths();
        }
    }
}

/**
 * Perform forward propegation.
 * 
 * TODO currently implementation 
 * has unnessesary copying of vectors.
 */
void cpu::NeuralNetwork::forward_propegation(cpu::Vector& x){

    this->m_hidden_layer1.forwardPropegation(x);
    cpu::Vector a1 = this->m_hidden_layer1.m_a;

    this->m_hidden_layer2.forwardPropegation(a1);
    cpu::Vector a2 = this->m_hidden_layer2.m_a;

    this->m_output_layer.forwardPropegation(a2);
}

  

/**
 * Make perdictions from the given z-score standardized test data.
 * The perdictions are made as follows.
 *      1. Iterate through the dataset.
 *      2. For each sample in the test set, perform forward propegation.
 *      3. Take the output m_a3 of the sigmoid activivation from the last layer
 *         can convert the value to 0 or 1 based of the given threeshold
 *         which will the perdiction for that given sample.
 *      4. Store the prediction in y_pred array. 
 * 
 * @param X_test_stand z-score standardized test data.
 * @param threeshold If m_a2 > threeshold, then the prediction is 1.
 *                   Otherwise, the prediction is 0.
 * 
 * @return A vector with predictions for each corresponding sample of
 *         X_test_stand.
 * 
 */
std::vector<double> cpu::NeuralNetwork::perdict( cpu::Dataset& X_test_stand, const double& threeshold){
    
    std::vector<double> y_pred(X_test_stand.get_num_rows());

    cpu::Vector x(3, 0.0);

    double a;


    for (int j=0; j < X_test_stand.get_num_rows(); j++){
        x = X_test_stand.getRow(j);

        this->forward_propegation(x);

        a = this->m_output_layer.m_a; 

        if(a > threeshold){
            y_pred[j] = 1.0;
        }else{
            y_pred[j] = 0.0;
        }
    }

    return y_pred;
}

/**
 * 
 * Compute the accuracy of the model using the following formula
 * @f$accuracy = \frac{correct predictions}{ total number of perdictions}$
 * 
 * @param y_pred The vector holding the perdictions of the model for every test sample
 * @param y_test The vector holding the true outcome for every test sample.
 * 
 * Assumption: y_pred and y_test are of the same size.
 * 
 * @return The accuracy score.
 * 
 */
double cpu::NeuralNetwork::computeAccuracy(std::vector<double>& y_pred, std::vector<double>& y_test){
    double accuracy = 0.0;

    for (int j=0; j < y_test.size(); j++){
        if(y_pred[j] == y_test[j]){
            accuracy++;
        }
    }

    accuracy = accuracy/static_cast<double>(y_test.size());

    return accuracy;
}

/**
 * 
 * Compute the loss of the neural network using the 
 * Cross-Entropy loss function.
 * 
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 * 
 * @param y The actual outcomes.
 * @param a The output of the sigmoid activation neuron.
 * 
 * @return entropy loss 
 * 
 * Assumptions: The values of activations are greater than zero since they are 
 * the result of sigmoid activation.
 * 
 */
double cpu::NeuralNetwork::bceLoss(const double &y, 
                                    const double &a){
    double loss = 0.0f;
    // Use epsilon since log of zero is undefined.
    double epsilon = 0.0001; 


    loss += -y*std::log(a + epsilon) - (1-y)*std::log(1-a + epsilon);



    return loss;
}

/**
 * Perform back propegation
 */
void cpu::NeuralNetwork::backPropegation(const cpu::Vector& x, const double& y){
    cpu::Vector a2 = this->m_hidden_layer2.m_a;
    this->m_output_layer.backPropegation(y, a2);

    cpu::Vector a1 = this->m_hidden_layer1.m_a;
    cpu::Vector W3 = this->m_output_layer.m_W;
    double delta3 = this->m_output_layer.m_delta;
    this->m_hidden_layer2.backPropegation(W3, delta3, a1);

    cpu::Matrix W2 = this->m_hidden_layer2.m_W;
    cpu::Vector delta2 = this->m_hidden_layer2.m_delta;
    this->m_hidden_layer1.backPropegation(W2, delta2,x);

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void cpu::NeuralNetwork::updateWeigths(){
    double alpha = this->m_alpha;
    this->m_output_layer.updateWeigths(alpha);
    this->m_hidden_layer2.updateWeigths(alpha);
    this->m_hidden_layer1.updateWeigths(alpha);
}
