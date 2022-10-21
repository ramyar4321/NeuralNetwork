#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

/**
 * Initialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(int hidden_layer1_size,
                                  int hidden_layer2_size,
                                  int epoch,
                                  float alpha):
                                  m_x(3, 0.0),
                                  m_y(0.0),
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
void cpu::NeuralNetwork::fit(cpu::Dataset& X_train_stand, std::vector<float>& y_train){

    // Initialize the weigths of neural network
    this->m_hidden_layer1.weightInitialization();
    this->m_hidden_layer2.weightInitialization();
    this->m_output_layer.weightInitialization();
    

    for (int e =0; e< this->m_epoch; e++){
        for (int j=0; j < X_train_stand.get_num_rows(); j++){
            this->m_x = X_train_stand.getRow(j);
            this->m_y = y_train[j];

            this->forwardPropegation();
            this->backPropegation();
            this->updateWeigths();
        }
    }
}

/**
 * Perform forward propegation.
 * 
 * @return The activation of the output neuron.
 * 
 */
float cpu::NeuralNetwork::forwardPropegation(){

    // Initialize vector to store activation 
    // of hidden layers. Initial size of the vector is 
    // not important since it will automatically be resized.
    cpu::Vector a_hiddenlayer(1, 0.0);

    // Initialize value to store the activation of ouput neuron
    float a_outputlayer = 0;

    a_hiddenlayer = this->m_hidden_layer1.forwardPropegation(this->m_x);
    a_hiddenlayer = this->m_hidden_layer2.forwardPropegation(a_hiddenlayer);
    a_outputlayer = this->m_output_layer.forwardPropegation(a_hiddenlayer);

    return a_outputlayer;
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
std::vector<float> cpu::NeuralNetwork::perdict( cpu::Dataset& X_test_stand, const float& threeshold){
    
    std::vector<float> y_pred(X_test_stand.get_num_rows());

    float a;


    for (int j=0; j < X_test_stand.get_num_rows(); j++){
        this->m_x = X_test_stand.getRow(j);

        a = this->forwardPropegation();

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
float cpu::NeuralNetwork::computeAccuracy(std::vector<float>& y_pred, std::vector<float>& y_test){
    float accuracy = 0.0;

    for (int j=0; j < y_test.size(); j++){
        if(y_pred[j] == y_test[j]){
            accuracy++;
        }
    }

    accuracy = accuracy/static_cast<float>(y_test.size());

    return accuracy;
}


/**
 * Perform back propegation
 */
void cpu::NeuralNetwork::backPropegation(){

    
    // Variables to store information to be passed
    // between layers. Note, the dimensions of the vectors
    // and matrices are not important since they will be resized.
    cpu::Vector W_outputlayer(1,0.0);
    float delta_outputlayer;

    cpu::Vector a_hiddenlayer(1,0.0);
    cpu::Matrix W_hiddenlayer(1,1);
    cpu::Vector delta_hiddenlayer(1,0.0);


    a_hiddenlayer = this->m_hidden_layer2.a();
    delta_outputlayer = this->m_output_layer.backPropegation(this->m_y, a_hiddenlayer);

    a_hiddenlayer = this->m_hidden_layer1.a();
    W_outputlayer = this->m_output_layer.W();
    delta_hiddenlayer = this->m_hidden_layer2.backPropegation(W_outputlayer, delta_outputlayer, a_hiddenlayer);

    W_hiddenlayer = this->m_hidden_layer2.W();
    this->m_hidden_layer1.backPropegation(W_hiddenlayer, delta_hiddenlayer, this->m_x);

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void cpu::NeuralNetwork::updateWeigths(){
    float alpha = this->m_alpha;
    this->m_output_layer.updateWeigths(alpha);
    this->m_hidden_layer2.updateWeigths(alpha);
    this->m_hidden_layer1.updateWeigths(alpha);
}


// Setter methods

void cpu::NeuralNetwork::x(const cpu::Vector& x){
    this->m_x = x;
}

void cpu::NeuralNetwork::y(const float& y){
    this->m_y = y;
}