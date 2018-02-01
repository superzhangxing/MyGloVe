#include <vector>
#include <string>
#include <unordered_map>

class Glove{
public:
    struct CooccurrenceRecord{
        int w1,w2;
        double weight;
    };

    Glove();
    ~Glove();

    void initilize_parameters();
    void calculate_vocab_size(std::string vocab_file);
    void calculate_num_cooccurrence_record(std::string shuffle_cooccurrence_record_file);
    
    void train_glove();
    void glove_thread(int vid);
    void save_parameters();

private:
    int write_header;
    int use_unk_vec;
    int num_threads;
    int num_iter;
    int vector_size;
    int save_gradsq;
    int use_binary;
    int model;

    double eta;
    double alpha,x_max;

    std::string vocab_file;
    std::string cooccurrence_record_file;
    long long num_cooccur_record;
    long long vocab_size;

    std::string vector_file;
    std::string parameter_file;
    std::vector<long long> vec_num_per_thread;

    // used for sgd
    std::vector<double> vec_W;
    std::vector<double> vec_b;
    std::vector<double> vec_W_cooccur;
    std::vector<double> vec_b_cooccur;
    // used for adaptive sgd
    std::vector<double> vec_W_grad;
    std::vector<double> vec_W_cooccur_grad;
    std::vector<double> vec_b_grad;
    std::vector<double> vec_b_cooccur_grad;
    // used for adam sgd
    std::vector<double> vec_W_V;
    std::vector<double> vec_W_S;
    std::vector<double> vec_W_cooccur_V;
    std::vector<double> vec_W_cooccur_S;
    std::vector<double> vec_b_V;
    std::vector<double> vec_b_S;
    std::vector<double> vec_b_cooccur_V;
    std::vector<double> vec_b_cooccur_S;
    double beta1,beta2;

    std::vector<double> vec_cost;
    
    //a hash map for word to rank
    std::unordered_map<int,std::string> hash_vocab;

};