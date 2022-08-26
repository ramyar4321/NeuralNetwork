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
gpu::NeuralNetwork::NeuralNetwork(int input_size, int layer1_size, int layer2_size, 
                                  int output_size, int epoch, int alpha):
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
    this->m_fprime1 = std::shared_ptr<float>(new float[m_layer1_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_fprime2 = std::shared_ptr<float>(new float[m_layer2_size]{0},
                                        [&](float* ptr){ delete[] ptr; });

    this->m_x = std::shared_ptr<float>(new float[m_input_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_delta1 = std::shared_ptr<float>(new float[m_layer1_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
    this->m_delta2 = std::shared_ptr<float>(new float[m_layer2_size]{0},
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
void gpu::NeuralNetwork::initializeWeights(std::shared_ptr<float>& W, int layerI_size, int layerJ_size){
    std::mt19937 generator;
    double mean = 0.0f;
    double stddev = std::sqrt(1 / static_cast<double>(layerI_size) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (int j=0; j< layerJ_size; ++j) {
        for (int i=0; i< layerI_size; ++i) {
            W.get()[j*layerI_size+i] = normal(generator);
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

    for (int e =0; e < this->m_epoch; e++){
        for (int j=0; j < X_train_stand.get_num_rows(); j++){
            x = X_train_stand.getRow(j);
            this->x(x);
            this->forwardPropegation();
            this->backPropegation();
            this->updateWeigths();
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
                                         const int layerI_size,
                                         const int layerJ_size)
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
                                         const int layerI_size,
                                         const int layerJ_size){


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
                                         const int layerI_size)

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
std::vector<float> gpu::NeuralNetwork::perdict( gpu::Dataset& X_test_stand, const float& threeshold){
    
    std::vector<float> y_pred(X_test_stand.get_num_rows());
    std::vector<float> x;


    for (int j=0; j < X_test_stand.get_num_rows(); j++){
        x = X_test_stand.getRow(j);
        this->x(x);

        this->forwardPropegation();

        if(m_a3 > threeshold){
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
float gpu::NeuralNetwork::computeAccuracy(std::vector<float>& y_pred, std::vector<float>& y_test){
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
void gpu::NeuralNetwork::backPropegation(){
    this->computeDeltaInit(this->m_delta3, this->m_y, this->m_a3, this->m_z3);
    this->computeGradientInit(this->m_dLdW3, this->m_delta3, this->m_a2,
                              this->m_layer2_size, this->m_output_size);

    this->reluPrime(this->m_fprime2, this->m_z2, this->m_layer2_size);
    this->computeDelta(this->m_delta2, this->m_W3, this->m_delta3, 
                       this->m_fprime2, this->m_layer2_size, this->m_output_size);
    this->computeGradient(this->m_dLdW2, this->m_delta2, this->m_a1,
                           this->m_layer1_size, this->m_layer2_size);

    this->reluPrime(this->m_fprime1, this->m_z1, this->m_layer1_size);
    this->computeDelta(this->m_delta1, this->m_W2, this->m_delta2, 
                       this->m_fprime1, this->m_input_size, this->m_layer1_size);
    this->computeGradient(this->m_dLdW1, this->m_delta1, this->m_x,
                           this->m_input_size, this->m_layer1_size);

}

/**
 * Compute the derivative of binary cross entropy loss function
 * @f$\frac{\partial L}{\partial a} = - \frac{y}{a} + \fra{1-y}{1-a}$ where
 * @f$a$ is the output of the output neuron.
 * Since division by zero can cause numerical issues, a small value, called epsilon,
 * will be added to the denominator terms.
 * 
 * @param y The outcomes.
 * @param a The output of the sigmoid activation neuron.
 * 
 * @return The derivative of the cross entropy loss function with
 *         respect to the signmoid activation a, in other words, f'(z)
 */
double gpu::NeuralNetwork::bceLossPrime(const double &y, const double &a){

    double loss = 0.0;
    double epsilon = 0.0001;

    loss += -(y/(a+epsilon)) + ((1-y)/(1-a+epsilon));

    return loss;
}

/**
 * Compute the derivative of the sigmoid activiation which can be defined as
 * @f$\sigma^{'} = \sigma(1-\sigma)$.
 * 
 * @param z The output of the output neuron in the last layer.
 * 
 * @return a The sigmoid prime activation.
 */
double gpu::NeuralNetwork::sigmoidPrime(const double& z){

    double a = sigmoid(z)*(1.0 - sigmoid(z));

    return a;
}

/**
 * Compute activation of neuron i of layer I using the derivative of ReLu
 * activation function.
 * @f$\begin{math}
        f'(z_i)=\left\{
            \begin{array}{ll}
                0, & \mbox{if $x<0$}.\\
                1, & \mbox{if $x>0$}.
            \end{array}
        \right.
    \end{math}$
 * The derivative is undefined at z_i = 0 but it can be set to zero
 * in order to produce sparse vector.
 * 
 * @param fprime A vector containing f' for each neuron in layer I
 * @param z A vector that contains the output of each neuron in layer I
 * @param layerI_size The number of neurons in layer I
 * 
 */
void gpu::NeuralNetwork::reluPrime(std::shared_ptr<float>& fprime,
                                    const std::shared_ptr<float> &z,
                                    const int layerI_size){

    for (int i = 0; i < layerI_size; i++){
        if(z.get()[i] <= 0){
            fprime.get()[i] = 0;
        }else{
            fprime.get()[i] = 1;
        }
    }
}

/**
 * Compute the error term associated with the output neuron.
 * The error term is commonly referred to as delta and is defined as the following
 * @f$\delta = f'(z)\frac{\partial L}{\partial a} = $
 * @f$         \sigma^{'}(z) (- \frac{y}{a} + \fra{1-y}{1-a})$
 * 
 * This function serves as a helper function for computeGradientInit.
 * 
 * @param y The outcomes from the dataset
 * @param a The activation of the sigmoid neuron
 * @param z the output of the sigmoid neuron
 * 
 * @return The error term associated with the output neuron.
 */
void gpu::NeuralNetwork::computeDeltaInit(double& delta,
                                            const double& y,
                                            const double& a,
                                            const double& z){

    delta = sigmoidPrime(z) * bceLossPrime(y, a);
}

/**
 * For layers J < K, compute the error term associated with each neuron j of layer k.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * @f$f'$ is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{kj}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * @param delta A vector containing the error terms for neuron j in layer J
 * @param W A flattened 2d matrix containing the weigths between layer J and K
 * @param delta_ A vector comtaining the error terms of each neuron k of layer K
 * @param fprime A vector containing f' for each neuron in layer I
 * @param layerJ_size The number of neurons in layer J
 * @param layerK_size The number of neurons in layer K
 * 
 */
void gpu::NeuralNetwork::computeDelta(std::shared_ptr<float>& delta,
                                        const std::shared_ptr<float>& W, 
                                        const std::shared_ptr<float>& delta_,
                                        const std::shared_ptr<float>& fprime,
                                        const int layerJ_size,
                                        const int layerK_size){

    float temp;
    for(int j=0; j < layerJ_size; j++){
        temp = 0.0;
        for(int k=0; k < layerK_size; k++){
            temp += W.get()[k*layerJ_size+j]*delta_.get()[k];
        }
        delta.get()[j] = temp*fprime.get()[j];
    }


}

/**
 * For layers J < K, compute the error term associated with each neuron j of layer k.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * @f$f'$ is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{kj}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * @param delta A vector containing the error terms for neuron j in layer J
 * @param W A flattened 2d matrix containing the weigths between layer J and K
 * @param delta_ A vector comtaining the error terms of each neuron k of layer K
 * @param fprime A vector containing f' for each neuron in layer I
 * @param layerJ_size The number of neurons in layer J
 * @param layerK_size The number of neurons in layer K
 * 
 */
void gpu::NeuralNetwork::computeDelta(std::shared_ptr<float>& delta,
                                        const std::shared_ptr<float>& W, 
                                        const double& delta_,
                                        const std::shared_ptr<float>& fprime,
                                        const int layerJ_size,
                                        const int layerK_size){

    float temp;
    for(int j=0; j < layerJ_size; j++){
        temp = 0.0;
        for(int k=0; k < layerK_size; k++){
            temp += W.get()[k*layerJ_size+j]*delta_;
        }
        delta.get()[j] = temp*fprime.get()[j];
    }

}


/**
 * 
 * Compute the gradient for each weight for a 
 * given layer except the last layer of the neural network.
 * For layers I < J, the gradient for any given weight can be computed as follows.
 * @f$\frac{dL}{dw_{ji}} = a_i * \delta_{j}$ where @f$w_{ji}$ is the weight from
 * neuron i of layer I to neuron j of layer J, @f$a_i$ is the activation of neuron
 * i of layer I, and @f$\delta_{j}$ is the error term of neuron j of layer J.
 * 
 * @param dLdW  A flatten 2d matric containing the partial derivatives for each
 *              weight between layers I and J
 * @param detla A vector containing the error terms for each neuron of layer J
 * @param a     A vector comtaining the activation of each neuron of layer I
 * @param layerI_size The number of neurons in layer I
 * @param layerJ_size The number of neurons in Layer J
 * 
 * 
 */
void gpu::NeuralNetwork::computeGradient(std::shared_ptr<float>& dLdW,
                                        const std::shared_ptr<float>& delta,
                                        const std::shared_ptr<float>& a,
                                        const int layerI_size,
                                        const int layerJ_size){

    for (int j=0; j< layerJ_size; ++j) {
        for (int i=0; i< layerI_size; ++i) {
            dLdW.get()[j*layerI_size+i] = a.get()[i]*delta.get()[j];
        }
    }
}

/**
 * 
 * Compute the gradient of the weight between the second hidden layer
 * and the output layer.
 * @f$dL/dW = \delta a$
 * 
 * @param dLdW  A vector containing the partial derivative between 
 *              the second hidden layer and the output layer.
 * @param delta The delta term from the output neuron.
 * @param a     A vector containing the activation of the second hidden layer.
 * @param layerI_size The number of neurons in layer I
 * @param LayerJ_size The number of neurons in layer J
 * 
 * 
 */
void gpu::NeuralNetwork::computeGradientInit(std::shared_ptr<float>& dLdW,
                                                const double& delta,
                                                const std::shared_ptr<float>& a,
                                                const int layerI_size,
                                                const int layerJ_size){

    for (int j=0; j< layerJ_size; ++j) {
        for (int i=0; i< layerI_size; ++i) {
            dLdW.get()[j*layerI_size+i] = a.get()[i]*delta;
        }
    }

}

void gpu::NeuralNetwork::updateWeigths(){
    this->gradientDecent(this->m_W3, this->m_dLdW3, this->m_layer2_size, this->m_output_size);
    this->gradientDecent(this->m_W2, this->m_dLdW2, this->m_layer1_size, this->m_layer2_size);
    this->gradientDecent(this->m_W1, this->m_dLdW1, this->m_input_size, this->m_layer1_size);
}
void gpu::NeuralNetwork::gradientDecent(std::shared_ptr<float>& W,
                                         const std::shared_ptr<float>& dLdW,
                                         const int layerI_size,
                                         const int layerJ_size){

    for (int j=0; j< layerJ_size; ++j) {
        for (int i=0; i< layerI_size; ++i) {
            W.get()[j*layerI_size+i] = W.get()[j*layerI_size+i] - this->m_alpha*dLdW.get()[j*layerI_size+i];
        }
    }

}
            

void gpu::NeuralNetwork::x(const std::vector<float>& x){
    for(int i=0; i < this->m_input_size; i++){
        this->m_x.get()[i] = x[i];
    }
}