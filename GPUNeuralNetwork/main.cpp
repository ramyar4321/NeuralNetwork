#include "NeuralNetwork.hpp"
#include "Dataset.hpp"

int main(){

    gpu::Dataset dat(4, 306,0.90);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    gpu::Dataset x_train = dat.X_train_split();
    gpu::Dataset x_train_stand = x_train.standardizeMatrix();
    std::vector<float> y_train = dat.y_train_split();

    gpu::NeuralNetwork net(3,3, 3, 1, 0.001, 1);
    net.fit(x_train_stand);


    return 0;

}