#include "NeuralNetwork.hpp"
#include "random"
#include <iostream>

/**
 * Initialize neural network memeber variables.
 * 
 * Note, since the shared pointer memeber variables are 
 * not specified in the constructor initialzier list,
 * they will be default intialized to nullpointer.
 * allocateMemeory will be used to properly initialize 
 * the shared pointer variables.
 */
gpu::NeuralNetwork::NeuralNetwork(int input_size, int layer1_size, int output_size,
                                  int layer2_size, int epoch, int alpha):
                                  m_input_size(input_size),
                                  m_layer1_size(layer1_size),
                                  m_layer2_size(layer2_size),
                                  m_output_size(output_size),
                                  m_z3(0.0f),
                                  m_a3(0.0f),
                                  m_y(0.0f),
                                  m_delta3(0.0f),
                                  m_epoch(epoch),
                                  m_alpha(alpha)
{
    allocatedMemeory();
}

/**
 * Allocate memeory on heap to store memeber variables. 
 * Note, we are using shared pointer to store an array of floats
 * for any given memeber variable and we need to specify the deleter.
 */
void gpu::NeuralNetwork::allocatedMemeory(){

    this->m_z1 = std::shared_ptr<float>(new float[m_layer1_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_z2 = std::shared_ptr<float>(new float[m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });

    this->m_a1 = std::shared_ptr<float>(new float[m_layer1_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_a2 = std::shared_ptr<float>(new float[m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });

     this->m_x = std::shared_ptr<float>(new float[m_input_size]{0},
                                        [&](float* ptr){ delete[] ptr; });

    this->m_W1 = std::shared_ptr<float>(new float[m_input_size*m_layer1_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_W2 = std::shared_ptr<float>(new float[m_layer1_size*m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_W3 = std::shared_ptr<float>(new float[m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });

    this->m_dLdW1 = std::shared_ptr<float>(new float[m_input_size*m_layer1_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_dLdW2 = std::shared_ptr<float>(new float[m_layer1_size*m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_dLdW3 = std::shared_ptr<float>(new float[m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
}

/**
 * Initialize the weights to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 */
void gpu::NeuralNetwork::initializeWeights(std::shared_ptr<float>& W, int layer_I, int layer_J){
    std::mt19937 generator;
    double mean = 0.0f;
    double stddev = std::sqrt(1 / static_cast<double>(layer_I) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (int j=0; j< layer_J; ++j) {
        for (int i=0; i< layer_I; ++i) {
            W.get()[j*layer_I+i] = normal(generator);
        }
    }
}

/**
 * Use the training data to estimate the parameters of the model.
 * 
 * @param X_train_stand The X train dataset used to train the Neural Network.
 *                      The X train dataset is assumed to be z-score standardized.
 */
void gpu::NeuralNetwork::fit(Dataset& X_train_stand){
    this->initializeWeights(this->m_W1, this->m_input_size, this->m_layer1_size);
    this->initializeWeights(this->m_W2, this->m_layer1_size, this->m_layer2_size);
    this->initializeWeights(this->m_W3, this->m_layer2_size, this->m_output_size);
    std::vector<float> x;

    for (int e =0; e< this->m_epoch; e++){
        for (int j=0; j < X_train_stand.get_num_rows(); j++){
            x = X_train_stand.getRow(j);
            this->x(x);

            this->forwardPropegation();
            //this->backPropegation();
            //this->updateWeigths();
        }
    }
}

/**
 * Perform forward propegation.
 * 
 */
void gpu::NeuralNetwork::forwardPropegation(){
    this->computeOutputs(this->m_z1, this->m_W1, 
                         this->m_x, this->m_input_size, this->m_layer1_size);
    this->reluActivation(this->m_a1, this->m_z1, this->m_layer1_size);

    this->computeOutputs(this->m_z2, this->m_W2, this->m_a1,
                          this->m_layer1_size, this->m_layer2_size);
    this->reluActivation(this->m_a2, this->m_z2, this->m_layer2_size);

    this->computeOutputs(this->m_z3, this->m_W3, this->m_a2,
                         this->m_layer2_size, this->m_output_size);
    this->m_a3 = this->sigmoid(this->m_z3);
}


/**
 * For layers I < J, compute the output of each neuron j in layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 * 
 * @param W A flatten 2d matrix that contains the weigths 
 *          connecting the neurons of layer I to the neurons of layer J.
 * @param a The vector that contains the activations of each neuron in layer I
 * @param layerI_size Number of neurons in layer I
 * @param layerJ_size Number of neurons in layer J
 * 
 * @return z The vector that contains the output of each neuron in layer J
 * 
 */
void gpu::NeuralNetwork::computeOutputs(std::shared_ptr<float>& z, 
                                         const std::shared_ptr<float>& W, 
                                         const std::shared_ptr<float>& a,
                                         int layerI_size,
                                         int layerJ_size)
{

    double temp = 0.0f;

    for(int j = 0; j < layerJ_size; j++){
        for(int i=0; i < layerI_size; i++){
            temp += W.get()[j*layerI_size + i]*a.get()[i];
        }
        z.get()[j] = temp;
    }

}

/**
 * This function computes the output of the neuron
 * in the last layer of the neural network. 
 * The output for such a neuron can be computed as follows 
 * @f$z = \sum_{i}^I w_{i} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 * 
 * @param W A flatten 2d matrix that contains the weigths 
 *          connecting the neurons of the second hidden layer to the output neuron.
 * @param a The vector that contains the activations of each neuron in the second hidden layer
 * @param layerI_size Number of neurons in layer I
 * @param layerJ_size Number of neurons in layer J
 * 
 * @return z The output of the output neuron.
 * 
 */
void gpu::NeuralNetwork::computeOutputs(double& z, 
                                         const std::shared_ptr<float>& W, 
                                         const std::shared_ptr<float>& a,
                                         int layerI_size,
                                         int layerJ_size){


    double temp = 0.0f;

    for(int j = 0; j < layerJ_size; j++){
        for(int i=0; i < layerI_size; i++){
            temp += W.get()[j*layerI_size + i]*a.get()[i];
        }
        z = temp;
    }

}

/**
 * Compute the activation of each neuron i in layer I using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_i = max(0, z_i)$. This method will be called for
 * hidden layers of the neural network.
 * 
 * @param z The vector that contains the output of each neuron in layer I
 * 
 * @return a The vector that contains the activations of each neuron in layer I
 * 
 */
void gpu::NeuralNetwork::reluActivation(std::shared_ptr<float>& a,
                                         const std::shared_ptr<float>& z,
                                         int layerI_size)

{

    for (int i=0; i < layerI_size; i++) {
        if(z.get()[i] > 0.0f ){
            a.get()[i] = z.get()[i];
        }else{
            a.get()[i] = 0.0f;
        }
    } 
}

/**
 * Compute the sigmoid activation of the output neuron.
 * 
 * The sigmoid activation function for the output neuron can be defined as the following
 * @f$\sigma (z_3) = \frac{1}{1+ \exp (- z_3)} = \frac{\exp (z_3)}{1+ \exp ( z_3)}$. 
 * If z_3 is positive and its magnitude is too large, it can cause overflow when computing @f$\exp ( z_3)$ and
 * if z_3 is negative and its magnitude is too large, it can cause overflow when computing @f$\exp (- z_3)$.
 * In order to avoid numerical instability, let @f$\sigma (z_3) = \frac{1}{1+ \exp (- z_3)}$ for @f$ z_3 >=0 $
 * and let @f$\sigma (z_3) = \frac{\exp (z_3)}{1+ \exp ( z_3)}$ for @f$ z_3 < 0 $.
 * 
 * @see https://stackoverflow.com/questions/41800604/need-help-understanding-the-caffe-code-for-sigmoidcrossentropylosslayer-for-mult
 * @see https://stackoverflow.com/questions/40353672/caffe-sigmoidcrossentropyloss-layer-loss-function
 * 
 * @param z The output of the output neuron in the last layer.
 * 
 * @return a The activation of the output neuron.
 * 
 */
double gpu::NeuralNetwork::sigmoid(const double& z)
{
    double a = 0.0; 

    if (z >= 0.0f) {
        a = 1.0f / (1.0f + std::exp(-z));
    } else {
        a = std::exp(z) / (1.0f + std::exp(z));
    }


    return a;
}

void gpu::NeuralNetwork::x(const std::vector<float>& x){
    for(int i=0; i < this->m_input_size; i++){
        this->m_x.get()[i] = x[i];
    }
}