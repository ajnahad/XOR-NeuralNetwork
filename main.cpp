#include <cmath>
#include <fstream> 
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>
#include <yaml-cpp/yaml.h>

class Neuron{
    std::vector<float> weights;
    float bias;
    float output;
    float delta; //error term for backprop
    
    public:
    Neuron(std::vector<float> w, float b){
        weights = w;
        bias = b;
    }
    float activate(std::vector <float> inputs){
        //calculating weighted sum/dot product with bias
        float weightedSum = 0.0;
        for (int i = 0; i < inputs.size(); i++){
            weightedSum += inputs.at(i) * weights.at(i);
        }
        weightedSum += bias;
        output = 1 / (1 + exp(-weightedSum));
        return output;
    }
    void update(std::vector<float> inputs, float delta, float learningRate){
        //gradient descent
        for (int i = 0; i < weights.size(); i++){
            weights.at(i) = weights.at(i) + learningRate * delta * inputs.at(i);
        }
        bias = bias + learningRate * delta * 1;
    }
    float getOutput(){return output;}
    float getDelta(){return delta;}
    std::vector<float> getWeights(){return weights;}
    void setDelta(float d){delta = d;}
};   

class Layer{
    std::vector<Neuron> neurons;
    std::vector<float> outputs;

    public:
    Layer(YAML::Node layerConfig){
        for(auto neuronNode : layerConfig){
            std::vector<float> weights = neuronNode["weights"].as<std::vector<float>>();
            float bias = neuronNode["bias"].as<float>();
            neurons.push_back(Neuron(weights,bias));
        }
    }
    std::vector<float> forward(std::vector<float> inputs){
        outputs.clear();
        for (auto& neuron : neurons){
            outputs.push_back(neuron.activate(inputs));
        }
        return outputs;
    }
    void backward(std::vector<float>& expectedOutputs, float learningRate, std::vector<float> inputs){ //for output layer
        for (int i = 0; i < neurons.size(); i++){
            float delta = (outputs.at(i) - expectedOutputs.at(i)) * outputs.at(i) * (1 - outputs.at(i));
            neurons.at(i).update(inputs, delta, learningRate);
            neurons.at(i).setDelta(delta);
        }
    }
    void backward(Layer& nextLayer, std::vector<float> inputs, float learningRate){ //for hidden layer
        for (int i = 0; i < neurons.size(); i++){ //loops through current layer of neurons
            float sum = 0.0;
            for (auto& nextNeuron : nextLayer.getNeurons()){
                sum = sum + nextNeuron.getWeights()[i] * nextNeuron.getDelta(); //gets the value which the neuron at position i on the next layer has and multiplies with delta
            }
            float delta = outputs.at(i) * (1 - outputs.at(i)) * sum;
            neurons.at(i).update(inputs, delta, learningRate);
            neurons.at(i).setDelta(delta);
        }
    }
    std::vector<float> getOutputs(){return outputs;}
    std::vector<Neuron>& getNeurons(){return neurons;}
};

class NeuralNetwork{
    std::vector<Layer> layers;
    float learningRate;
    int epochs;
    std::vector<std::pair<std::vector<float>, float>> dataset;
    YAML::Node config = YAML::LoadFile("config.yaml");
    public:
    NeuralNetwork(){
        YAML::Node hiddenLayerConfig = config["hidden_layer"];
        YAML::Node outputLayerConfig = config["output_layer"];
        layers.push_back(Layer(config["hidden_layer"]));
        layers.push_back(Layer(config["output_layer"]));
        learningRate = config["learning_rate"].as<float>();
        epochs = config["epochs"].as<int>();
    }
    void loadDataset(){
        std::ifstream file("dataset.txt");
        std::string line;
        while(std::getline(file, line)){
            std::istringstream iss(line);
            float num1, num2, expected;
            iss >> num1 >> num2 >> expected;
            std::vector<float> inputs = {num1, num2};
            dataset.push_back({inputs, expected});
        }
        file.close();
    }
    void train(){
        for (int i = 0; i < epochs; i++){
            for (int j = 0; j < dataset.size(); j++){
                std::vector<float> inputs = dataset.at(j).first;
                float output = dataset.at(j).second;
                std::vector<std::vector<float>> inputsToLayer(layers.size() + 1);
                inputsToLayer.at(0) = inputs;
                for (int k = 0; k < layers.size(); k++){
                    std::vector<float> outputs = layers.at(k).forward(inputsToLayer.at(k));
                    inputsToLayer.at(k+1) = outputs;
                }
                layers.back().backward(std::vector<float>{output}, learningRate, inputsToLayer.back());
                for (int k = layers.size() - 2; k >= 0; k--){
                    layers.at(k).backward(layers.at(k+1), inputsToLayer.at(k+1), learningRate);
                }
            }
            float loss = 0.0;
                for (auto& data : dataset){
                    float prediction = predict(data.first);
                    float error = data.second - prediction;
                    loss += error * error;
                }
                std::cout << "Epoch " << i + 1 << "; Loss: " << loss / dataset.size() << "\n";
        }
    }
    float predict(std::vector<float> inputs){
        for (int i = 0; i < layers.size(); i++){
            inputs = layers.at(i).forward(inputs);
        }
        return inputs.at(0);
    }

    std::vector<std::pair<std::vector<float>, float>>& getDataset(){return dataset;}
};

int main(){
    NeuralNetwork neuralNet;
    neuralNet.loadDataset();
    neuralNet.train();
    for(auto& data: neuralNet.getDataset()){ 
        float prediction = neuralNet.predict(data.first);
        std::cout << "Input: "; for (auto value : data.first) {std::cout << value << " ";}; 
        std::cout << "Predicted: " << prediction << ", Expected: " << data.second << "\n";
    }
    return 0;
}
