#pragma GCC optimize("-Ofast,inline,omit-frame-pointer,unroll-loops")
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <chrono>
using namespace std;

chrono::high_resolution_clock::time_point now;
#define MARKTIME now = chrono::high_resolution_clock::now();
#define TIME chrono::duration_cast<chrono::duration<double>>(chrono::high_resolution_clock::now() - now).count()

namespace utils{
    static unsigned int g_seed;
    inline void fast_srand(int seed) { //Seed the generator
      g_seed = seed;
    }
    inline int fastrand() {  //fastrand routine returns one integer, similar output value range as C lib.
      g_seed = (214013*g_seed+2531011);
      return (g_seed>>16)&0x7FFF;
    }
    inline int fastRandRange(int maxSize) {
      return fastrand() % maxSize;
    }
    inline float fastRandFloat(float a, float b) {
      return a + (static_cast<float>(fastrand()) / 0x7FFF)*(b-a);
    }
    float get_approx(float z, int nb_decimal){
        float coeff = 1;
        for (int i=0; i<nb_decimal; i++)
            coeff *= 10;
        return round(z * coeff) / coeff;
    }
}
using namespace utils;

float LEARNING_RATE = 0.001;
float MOMENTUM = 0.9;
float ALPHA = 0.0001;
struct Neuron{
    float bias;
    vector<float> weights;
    int nb_weights;
    float z; // sortie avant non-linéarité
    float activation; // après non-linéarité
    float delta;
    float momentum_bias = 0.0;
    vector<float> momentum_weights;

    Neuron(){}

    Neuron(const vector<Neuron> &prev_layer_neurons, float init_bound){
        this->nb_weights = prev_layer_neurons.size();
        this->bias = fastRandFloat(-init_bound, init_bound);
        for (int i=0; i<nb_weights; i++)
            this->weights.push_back(fastRandFloat(-init_bound, init_bound));
        this->momentum_weights.resize(nb_weights, 0.0);
    }

    void backprop(const vector<Neuron> &prev_layer_neurons){
        //if (delta < 1e-10 && delta > -1e-10)
        //    return;

        if (MOMENTUM > 0){
            momentum_bias = MOMENTUM * momentum_bias + LEARNING_RATE * (delta + ALPHA * bias);
            bias -= momentum_bias;
            for (int j=0; j<nb_weights; j++){
                float &mom = momentum_weights[j];
                mom = MOMENTUM * mom + LEARNING_RATE * (prev_layer_neurons[j].activation * delta + ALPHA * weights[j]);
                weights[j] -= mom;
            }
        }
        else{
            bias -= LEARNING_RATE * (delta + ALPHA * bias);
            for (int j=0; j<nb_weights; j++)
                weights[j] -= LEARNING_RATE * (prev_layer_neurons[j].activation * delta + ALPHA * weights[j]);
        }
    }

    friend ostream& operator<< (ostream &out, const Neuron &n){
        out << "\t weights ";
        for (float w : n.weights)
            out << get_approx(w, 4) << " ";
        float bias = get_approx(n.bias, 4);
        float z = get_approx(n.z, 4);
        float a = get_approx(n.activation, 4);
        float delta = get_approx(n.delta, 4);
        out << "bias " << bias << " z " << z << " a " << a << " delta " << delta;
        return out;
    }
};

enum LAYER_TYPE{INPUT, HIDDEN, OUTPUT};
struct Layer{
    vector<Neuron> neurons;
    int nb_neurons;
    LAYER_TYPE layer_type;

    Layer(int nb_neurons) : nb_neurons(nb_neurons){
        // premiere couche
        layer_type = INPUT;
        Neuron basic_neuron;
        neurons.resize(nb_neurons, basic_neuron);
    }

    Layer(int nb_neurons, Layer &prev_layer, LAYER_TYPE layer_type): nb_neurons(nb_neurons), layer_type(layer_type){
        float init_bound = sqrt(6. / float(nb_neurons + prev_layer.nb_neurons));
        for (int i=0; i<nb_neurons; i++)
            neurons.push_back(Neuron(prev_layer.neurons, init_bound));
    }

    float get_activation(float z){
        if (layer_type == HIDDEN)
            return (z > 0 ? z : 0);
        else if (layer_type == OUTPUT)
            return tanh(z);
        cerr << "ERROR : activation";
        return -1;
    }

    float get_derivative_activation(float z){
        if (layer_type == HIDDEN)
            return (z > 0 ? 1 : 0);
        else if (layer_type == OUTPUT){
            float z_tan = tanh(z);
            return 1 - z_tan * z_tan;
        }
        cerr << "ERROR : derivative_activation";
        return -1;
    }


    void feedforward(const vector<float> &input_vals){
        // premiere couche
        for (int i=0; i<nb_neurons; i++)
            neurons[i].activation = input_vals[i];
    }

    void feedforward(const Layer &prev_layer){
        int nb_weights = prev_layer.nb_neurons;
        for (int i=0; i<nb_neurons; i++){
            Neuron* neuron = &neurons[i];
            float z = 0.0;
            for (int j=0; j<nb_weights; j++)
                z += neuron->weights[j] * prev_layer.neurons[j].activation;
            z += neuron->bias;
            neuron->z = z;
            neuron->activation = get_activation(z);
        }
    }

    void backprop(const Layer &prev_layer, const vector<float> &targets){
        // dernière couche
        for (int i=0; i<nb_neurons; i++){
            Neuron* neuron = &neurons[i];
            float dz = get_derivative_activation(neuron->z);
            neuron->delta = (neuron->activation - targets[i]) * dz;
            neuron->backprop(prev_layer.neurons);
        }
    }

    void backprop(const Layer &prev_layer, const Layer &next_layer){
        // couches cachées
        for (int i=0; i<nb_neurons; i++){
            Neuron* neuron = &neurons[i];
            float dz = get_derivative_activation(neuron->z);
            float sum = 0.0;
            for (int j=0; j<next_layer.nb_neurons; j++)
                sum += next_layer.neurons[j].delta * next_layer.neurons[j].weights[i];
            neuron->delta = sum * dz;
            neuron->backprop(prev_layer.neurons);
        }
    }

    friend ostream& operator<< (ostream &out, const Layer &l){
        for (const Neuron &neuron : l.neurons)
            out << neuron << endl;
        return out;
    }
};

int NB_ITER = 100;
struct Neural_Network{
    vector<Layer> layers;
    int nb_layers;

    Neural_Network(vector<int> topology){
        nb_layers = topology.size();
        layers.push_back(Layer(topology[0]));
        for (int i=1; i<nb_layers; i++){
            LAYER_TYPE layer_type = (i == (nb_layers - 1) ?  OUTPUT : HIDDEN);
            layers.push_back(Layer(topology[i], layers[i-1], layer_type));
        }
    }

    void feedForward(const vector<float> &inputVals){
        layers[0].feedforward(inputVals);
        for (int i=1; i<nb_layers; i++){
            layers[i].feedforward(layers[i-1]);
        }
    }

    void backprop(const vector<float> &targets){
        layers[nb_layers-1].backprop(layers[nb_layers-2], targets);
        for (int i=nb_layers-2; i>0; i--){
            layers[i].backprop(layers[i-1], layers[i+1]);
        }
    }

    vector<float> predict(const vector<float> &inputVals){
        feedForward(inputVals);
        vector<float> outputs;
        Layer* output_layer = &layers.back();
        for (Neuron &neuron : output_layer->neurons)
            outputs.push_back(neuron.activation);
        return outputs;
    }

    float get_loss(const vector<float> &targets){
        float loss = 0.;
        Layer &output_layer = layers.back();
        for (int i=0; i<output_layer.nb_neurons; i++){
            float delta = output_layer.neurons[i].activation - targets[i];
            loss += 0.5 * delta * delta;
        }
        return loss;
    }

    void fit(const vector<vector<float>> &data_inputs, const vector<vector<float>> &data_targets){
        int BATCH_SIZE = max(1000, (int)data_inputs.size());

        int data_size = data_inputs.size();
        for (int iter=0; iter<NB_ITER; iter++){
            float loss = 0.;
            for (int j=0; j<BATCH_SIZE; j++){
                int rnd_i = fastRandRange(data_size);
                feedForward(data_inputs[rnd_i]);
                loss += get_loss(data_targets[rnd_i]);
                backprop(data_targets[rnd_i]);
            }
            float mean_loss = loss / float(BATCH_SIZE);
            cerr << "Iteration " << iter << ", mean_loss " << mean_loss << endl;
        }
    }

    friend ostream& operator<< (ostream &out, const Neural_Network &nn){
        for (int i=1; i<nn.nb_layers; i++)
            cout << "layer " << i << " :" << endl << nn.layers[i];
        return out;
    }
};

struct Data_Set{
    int nb_inputs;
    int nb_targets;
    vector<vector<float>> pool_inputs;
    vector<vector<float>> pool_targets;
    int pool_size;
    vector<vector<float>> training_inputs;
    vector<vector<float>> training_targets;
    int training_size;
    vector<vector<float>> test_inputs;
    vector<vector<float>> test_targets;
    int test_size;

    Data_Set(string file, int nb_inputs, int nb_targets, float training_part = 0.85){
        this->nb_inputs = nb_inputs;
        this->nb_targets = nb_targets;
        read_file(file);
        scale(pool_inputs, -1., 1.);
        scale(pool_targets, -1., 1.);
        shuffle_data();
        split_data(training_part);
    }

    void read_file(string file){
        ifstream data_file = ifstream(file, ios::in);
        while(!data_file.eof()){
            vector<float> inputs(nb_inputs);
            vector<float> targets(nb_targets);
            for (int i=0; i<nb_inputs; i++)
                data_file >> inputs[i];
            for (int i=0; i<nb_targets; i++)
                data_file >> targets[i];
            pool_inputs.push_back(inputs);
            pool_targets.push_back(targets);
        }
        data_file.close();
        pool_size = pool_inputs.size();
    }

    void scale(vector<vector<float>> &to_scale, float lower, float upper){
        int vector_size = to_scale[0].size();
        vector<float> max_inputs;
        vector<float> min_inputs;
        max_inputs.resize(vector_size, -INFINITY);
        min_inputs.resize(vector_size, INFINITY);
        for (vector<float> &inputs : to_scale){
            for (int i=0; i<vector_size; i++){
                if (inputs[i] > max_inputs[i])
                    max_inputs[i] = inputs[i];
                else if (inputs[i] < min_inputs[i])
                    min_inputs[i] = inputs[i];
            }
        }

        for (vector<float> &inputs : to_scale){
            for (int i=0; i<vector_size; i++){
                inputs[i] = lower + (upper - lower) * (inputs[i] - min_inputs[i]) / (max_inputs[i] - min_inputs[i]);
            }
        }
    }

    void split_data(float training_part){
        training_size = training_part * pool_size;
        test_size = pool_size - training_size;
        for (int i=0; i<pool_size; i++){
            if (i < training_size){
                training_inputs.push_back(pool_inputs[i]);
                training_targets.push_back(pool_targets[i]);
            }
            else{
                test_inputs.push_back(pool_inputs[i]);
                test_targets.push_back(pool_targets[i]);
            }
        }
    }

    void shuffle_data(){
        vector<float> indices(pool_size);
        vector<float> rnd_vals(pool_size);
        for (int i=0; i<pool_size; i++){
            indices[i] = i;
            rnd_vals[i] = fastRandFloat(0, 1);
        }
        sort(indices.begin(), indices.end(), [rnd_vals](float a, float b){ return rnd_vals[a] < rnd_vals[b];});
        vector<vector<float>> temp_inputs(pool_size);
        vector<vector<float>> temp_outputs(pool_size);
        for (int i=0; i<pool_size; i++){
            temp_inputs[i] = pool_inputs[indices[i]];
            temp_outputs[i] = pool_targets[indices[i]];
        }
        pool_inputs = temp_inputs;
        pool_targets = temp_outputs;
    }

    static float classification_score(Neural_Network &NN, const vector<vector<float>> &data_inputs, const vector<vector<float>> &data_targets){
        int nb_ok = 0;
        int data_size = data_inputs.size();
        for (int i=0; i<data_size; i++){
            vector<float> inputs = data_inputs[i];
            vector<float> targets = data_targets[i];
            vector<float> outputs = NN.predict(inputs);
            int target_class = max_element(targets.begin(), targets.end()) - targets.begin();
            int output_class = max_element(outputs.begin(), outputs.end()) - outputs.begin();
            bool well_classified = (output_class == target_class);
            if (well_classified)
                nb_ok++;
            /*if (i == 0){
                cerr << "inputs: ";
                Data_Set::cerr_vector(inputs);
                cerr << "targets: ";
                Data_Set::cerr_vector(targets);
                cerr << "outputs: ";
                Data_Set::cerr_vector(outputs);
                cerr << " ok: " << well_classified << endl;
            }*/
        }
        return nb_ok / float(data_size);
    }

    template<class T>
    static void cerr_vector(const vector<T> vect, int nb_decimals = 3){
        for (float val : vect)
            cerr << get_approx(val, nb_decimals) << " ";
    }
};

int main(){
    fast_srand(time(NULL));

    //------------------------------------------------
    // parameters
    //------------------------------------------------
    string data_file_name = "iris_plant.dat"; // classification test case : https://en.wikipedia.org/wiki/Iris_flower_data_set
    int nb_inputs = 4;
    int nb_outputs = 3;
    float training_part = 0.75; // 75% of data for training, 25 % for test

    vector<int> NN_topology = {nb_inputs, 16, 16, nb_outputs}; // 2 hidden layers
    LEARNING_RATE = 0.0001; // Learning rate schedule for weight updates.
    MOMENTUM = 0.9; // Momentum for gradient descent update. Should be between 0 and 1 (put 0 to disable).
    ALPHA = 0.0001; // L2 penalty (regularization term) parameter (put 0 to disable).
    NB_ITER = 50;
    //------------------------------------------------
    //------------------------------------------------


    //------------------------------------------------
    // parameters
    //------------------------------------------------
    /*string data_file_name = "titanic.dat"; // classification test case : https://www.kaggle.com/c/titanic
    int nb_inputs = 7;
    int nb_outputs = 2; // survivor, dead
    float training_part = 0.75; // 75% of data for training, 25 % for test

    vector<int> NN_topology = {nb_inputs, 100, nb_outputs}; // 1 hidden layers
    LEARNING_RATE = 0.001; // Learning rate schedule for weight updates.
    MOMENTUM = 0.9; // Momentum for gradient descent update. Should be between 0 and 1 (put 0 to disable).
    ALPHA = 0.001; // L2 penalty (regularization term) parameter (put 0 to disable).
    NB_ITER = 100;*/
    //------------------------------------------------
    //------------------------------------------------

    // read data
    Data_Set data(data_file_name, nb_inputs, nb_outputs, training_part);

    // construct Neural network
    Neural_Network NN(NN_topology);

    // training
    MARKTIME
    NN.fit(data.training_inputs, data.training_targets);
    cerr << "training time : " << TIME << " seconds " << endl;
    float training_score = Data_Set::classification_score(NN, data.training_inputs, data.training_targets);
    cerr << "success rate (training data) : " << training_score * 100 << " % training_size : " << data.training_size << endl;

    // test
    float test_score = Data_Set::classification_score(NN, data.test_inputs, data.test_targets);
    cerr << "success rate (test data) : " << test_score * 100 << " % test_size : " << data.test_size << endl;

    //cerr << NN << endl;
}
