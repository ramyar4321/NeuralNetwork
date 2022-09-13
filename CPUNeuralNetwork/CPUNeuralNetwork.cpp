#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"
#include "Layers/Layer.hpp"

/**
 * Initialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(int epoch,
                                  double alpha):
                                  m_x(3, 0.0),
                                  m_y(1, 0.0),
                                  m_num_layers(0),
                                  m_epoch(epoch),
                                  m_alpha(alpha)
{}

/**
 * 
 * Add layers to the neural network.
 * 
 */
void cpu::NeuralNetwork::addLayer(cpu::Layer* layer){
    this->m_layers.push_back(layer);
    m_num_layers++;
}



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
    this->weightInitialization();
    

    for (int e =0; e< this->m_epoch; e++){
        for (int j=0; j < X_train_stand.get_num_rows(); j++){
            this->m_x = X_train_stand.getRow(j);
            this->m_y[0] = y_train[j];

            this->forwardPropegation();
            this->backPropegation();
            this->updateWeigths();
        }
    }
}

void cpu::NeuralNetwork::weightInitialization(){
    for(int l = 0; l < this->m_num_layers; l++){
        this->m_layers[l]->weightInitialization();
    }
}

/**
 * Perform forward propegation.
 * 
 * @return A vector containing the activation of each neuron
 *         in the output layer.
 * 
 */
cpu::Vector cpu::NeuralNetwork::forwardPropegation(){

    cpu::Vector z = this->m_x;
    for(int l = 0; l < m_num_layers; l++){
        z = m_layers[l]->forwardPropegation(z);
    }

    return z;

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

    //cpu::Vector x(3, 0.0);

    cpu::Vector a3(1, 0.0);


    for (int j=0; j < X_test_stand.get_num_rows(); j++){
        this->m_x = X_test_stand.getRow(j);

        a3 = this->forwardPropegation();

        if(a3[0] > threeshold){
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
 * Perform back propegation
 */
void cpu::NeuralNetwork::backPropegation(){

    // Three variables used between layers, namely
    // weights W, activations a, and error terms delta.
    // 
    cpu::Matrix W(1, 10);
    cpu::Vector delta(1, 0.0);
    cpu::Vector a(10, 0.0);

    for(int l = this->m_num_layers - 1; l >= 0; l--){
        if(l = this->m_num_layers-1){
            delta = this->m_layers[l]->backPropegation(m_y, a);
            W = this->m_layers[l]->W();
            a = this->m_layers[l-1]->a();
        }else{

        }if(l= 0){
            delta = this->m_layers[l]->backPropegation(W, delta, this->m_x);
        }else{
            delta = this->m_layers[l]->backPropegation(W, delta, a);
            W = this->m_layers[l]->W();
            a = this->m_layers[l-1]->a();
        }

    }

    /*
    cpu::Vector a2 = this->m_hidden_layer2.m_a;
    this->m_output_layer.backPropegation(this->m_y, a2);

    cpu::Vector a1 = this->m_hidden_layer1.m_a;
    cpu::Vector W3 = this->m_output_layer.m_W;
    double delta3 = this->m_output_layer.m_delta;
    this->m_hidden_layer2.backPropegation(W3, delta3, a1);

    cpu::Matrix W2 = this->m_hidden_layer2.m_W;
    cpu::Vector delta2 = this->m_hidden_layer2.m_delta;
    this->m_hidden_layer1.backPropegation(W2, delta2, this->m_x);*/

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void cpu::NeuralNetwork::updateWeigths(){
    double alpha = this->m_alpha;

    for(int l = 0;  l < this->m_num_layers; l++){
        this->m_layers[l]->updateWeigths(this->m_alpha);
    }
}

/**
 * Set the input of the neural network.
 */
void cpu::NeuralNetwork::x(const cpu::Vector& x){
    this->m_x = x;
}

/** 
 * Set weights for given layer of the neural network.
 */
void cpu::NeuralNetwork::W(const cpu::Matrix& W, const int& layer_index){
    this->m_layers[layer_index]->W(W);
}
