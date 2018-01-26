#include "Glove.h"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ratio>
#include <ctime>
#include <thread>
#include <assert.h>
#include <math.h>
#include <unordered_map>

double check_nan(double update) {
    if (isnan(update) || isinf(update)) {
        std::cerr<<"caught nan or inf in update"<<std::endl;
        return 0.;
    } else {
        return update;
    }
}

Glove::Glove(){
    this->initilize_parameters();
}

Glove::~Glove(){

}

void Glove::initilize_parameters(){
    write_header = 0;
    use_unk_vec = 1;
    num_threads = 4;//8
    num_iter = 15;
    vector_size = 50;
    save_gradsq = 0;
    use_binary = 0;
    model = 2;

    eta = 0.05;
    alpha = 0.75;
    x_max = 10;

    vocab_file = std::string("vocab.txt");
    cooccurrence_record_file = std::string("cooccur_shuffle");
    //cooccurrence_record_file = std::string("cooccur");
    calculate_vocab_size(vocab_file);
    calculate_num_cooccurrence_record(cooccurrence_record_file);

    parameter_file = std::string("");
    vector_file = std::string("word_vector.txt");

    vec_num_per_thread.resize(num_threads);

    vec_cost.resize(num_threads,0.0);

    vec_W.resize(vector_size*vocab_size);
    vec_W_cooccur.resize(vector_size*vocab_size);
    vec_b.resize(vocab_size);
    vec_b_cooccur.resize(vocab_size);
    vec_W_grad.resize(vector_size*vocab_size);
    vec_W_cooccur_grad.resize(vector_size*vocab_size);
    vec_b_grad.resize(vocab_size);
    vec_b_cooccur_grad.resize(vocab_size);
    for(int i=0;i<vector_size*vocab_size;++i){
        // vec_W[i] = 0.01;
        // vec_W_cooccur[i] = 0.01;
        vec_W[i] = (rand() / (double)RAND_MAX - 0.5) / vector_size;
        vec_W_cooccur[i] = (rand() / (double)RAND_MAX - 0.5) / vector_size;
        vec_W_grad[i] = 1.;
        vec_W_cooccur_grad[i] = 1.;
    }
    for(int i=0; i<vocab_size; ++i){
        // vec_b[i] = 0;
        // vec_b_cooccur[i] = 0;
        vec_b[i] = (rand() / (double)RAND_MAX - 0.5) / vector_size;
        vec_b_cooccur[i] = (rand() / (double)RAND_MAX - 0.5) / vector_size;
        vec_b_grad[i] = 1.;
        vec_b_cooccur_grad[i] = 1.;
    }
}

void Glove::calculate_vocab_size(std::string vocab_file){
    std::ifstream in(vocab_file);
    if(!in.is_open()){
        std::cerr << "error open vocab file!"<< std::endl;
        return;
    }

    std::string word;
    int frequency;
    int rank = 0;

    // bool good = in.good();
    // bool eof = in.eof();
    // bool fail = in.fail();
    // bool bad = in.fail();

    while(in.good()){
        in >> word >> frequency;
        std::pair<int,std::string> temp_pair(rank,word);
        hash_vocab.insert(temp_pair);
        ++rank;
    }
    vocab_size = hash_vocab.size();

    in.close();
}

void Glove::calculate_num_cooccurrence_record(std::string shuffle_cooccurrence_record_file){
    std::ifstream in(shuffle_cooccurrence_record_file,std::ifstream::binary);
    if(!in.is_open()){
        std::cerr<<"error open shuffle cooccurrence record file!"<<std::endl;
        return;
    }

    in.seekg(0,in.end);
    long long length = in.tellg();
    num_cooccur_record = length/sizeof(CooccurrenceRecord);

    in.close();
}

void Glove::train_glove(){
    using std::chrono::system_clock;

    system_clock::time_point current_time = system_clock::now();
    std::time_t tt;
    tt = system_clock::to_time_t(current_time);
    std::cerr << std::string("start train glove: ")<<ctime(&tt)<<std::endl;

    assert(num_threads > 0);
    assert(num_iter > 0);

    // allocate record numbers to all threads
    for(int i=0;i<num_threads;++i){
        vec_num_per_thread[i] = num_cooccur_record/num_threads;
    }
    vec_num_per_thread[num_threads-1] = num_cooccur_record - num_cooccur_record/num_threads*(num_threads-1);

    for(int i=0;i<num_iter;++i){
        double total_cost = 0.0;
        std::vector<std::thread> vec_thread;
        for(int i=0;i<num_threads;++i) vec_thread.push_back(std::thread(&Glove::glove_thread,this,i));
        for(int i=0;i<num_threads;++i) vec_thread[i].join();
        for(int i=0;i<num_threads;++i) total_cost += vec_cost[i];

        current_time = system_clock::now();
        tt = system_clock::to_time_t(current_time);
        std::cerr<<ctime(&tt)<<"   "<<"cost: "<<total_cost/num_cooccur_record<<std::endl;

    }
}

//train with adaptive sgd
void Glove::glove_thread(int vid){
    long long a,b,l1,l2;
    double diff,fdiff,temp1,temp2;
    CooccurrenceRecord record;

    std::ifstream in(cooccurrence_record_file,std::ifstream::binary);
    if(!in.is_open()){
        std::cerr << "error open file! glove thread "<<vid<<std::endl;
        return;
    }
    in.seekg((num_cooccur_record/num_threads*vid)*(sizeof(CooccurrenceRecord)),in.beg);

    vec_cost[vid] = 0.0;
    std::vector<double> vec_W_update(vector_size);
    std::vector<double> vec_W_cooccur_update(vector_size);
    
    for(a=0; a<vec_num_per_thread[vid];++a){
        in.read((char *)&record, sizeof(CooccurrenceRecord));
        
        // offset location of w1,w2 in vec_W
        l1 = record.w1*vector_size;
        l2 = record.w2*vector_size;

        //calculate cost
        diff = 0.;
        for(b=0;b<vector_size;++b) diff += vec_W[b+l1] * vec_W_cooccur[b+l2];
        diff += vec_b[record.w1] + vec_b[record.w2] - log(record.weight);
        fdiff = (record.weight > x_max) ? diff : pow(record.weight/x_max,alpha) *diff;

        //check nan and inf
        if(isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)){
            std::cerr<<"caught nan or inf in diff or fdiff for thread,skip update"<<std::endl;
            continue;
        }

        vec_cost[vid] += 0.5 * fdiff * diff;

        // adaptive gradient updates
        fdiff *= eta;
        double W_update_sum = 0;
        double W_cooccur_update_sum = 0;
        for(b=0; b < vector_size; ++b){
            temp1 = fdiff * vec_W_cooccur[b+l2];
            temp2 = fdiff * vec_W[b+l1];

            vec_W_update[b] = temp1 / sqrt(vec_W_grad[b+l1]);
            vec_W_cooccur_update[b] = temp2 / sqrt(vec_W_cooccur_grad[b+l2]);
            W_update_sum += vec_W_update[b];
            W_cooccur_update_sum += vec_W_cooccur_update[b];
            vec_W_grad[b+l1] += temp1 * temp1;
            vec_W_cooccur_grad[b+l2] += temp2 * temp2;
        }

        if(!isnan(W_update_sum) && !isinf(W_update_sum) && !isnan(W_cooccur_update_sum)
            && !isinf(W_cooccur_update_sum)){
            for(b=0; b<vector_size; ++b){
                vec_W[b+l1] -= vec_W_update[b];
                vec_W_cooccur[b+l2] -= vec_W_cooccur_update[b];
            }
        }

        vec_b[record.w1] -= check_nan(fdiff / sqrt(vec_b_grad[record.w1]));
        vec_b_cooccur[record.w2] -= check_nan(fdiff/sqrt(vec_b_cooccur_grad[record.w2]));
        fdiff *= fdiff;
        vec_b_grad[record.w1] += fdiff;
        vec_b_cooccur_grad[record.w2] += fdiff;
    }

}

// train with navie sgd
// void Glove::glove_thread(int vid){
//     long long a,b,l1,l2;
//     double diff,fdiff,temp1,temp2;
//     CooccurrenceRecord record;

//     std::ifstream in(cooccurrence_record_file,std::ifstream::binary);
//     if(!in.is_open()){
//         std::cerr << "error open file! glove thread "<<vid<<std::endl;
//         return;
//     }
//     in.seekg((num_cooccur_record/num_threads*vid)*(sizeof(CooccurrenceRecord)),in.beg);

//     vec_cost[vid] = 0.0;
//     std::vector<double> vec_W_update(vector_size);
//     std::vector<double> vec_W_cooccur_update(vector_size);
    
//     for(a=0; a<vec_num_per_thread[vid];++a){
//         in.read((char *)&record, sizeof(CooccurrenceRecord));
        
//         // offset location of w1,w2 in vec_W
//         l1 = record.w1*vector_size;
//         l2 = record.w2*vector_size;

//         //calculate cost
//         diff = 0.;
//         for(b=0;b<vector_size;++b) diff += vec_W[b+l1] * vec_W_cooccur[b+l2];
//         diff += vec_b[record.w1] + vec_b[record.w2] - log(record.weight);
//         fdiff = (record.weight > x_max) ? diff : pow(record.weight/x_max,alpha) *diff;

//         //check nan and inf
//         if(isnan(diff) || isnan(fdiff) || isinf(diff) || isinf(fdiff)){
//             std::cerr<<"caught nan or inf in diff or fdiff for thread,skip update"<<std::endl;
//             continue;
//         }

//         vec_cost[vid] += 0.5 * fdiff * diff;

//         // adaptive gradient updates
//         double W_update_sum = 0;
//         double W_cooccur_update_sum = 0;
//         for(b=0; b < vector_size; ++b){
//             temp1 = fdiff * vec_W_cooccur[b+l2];
//             temp2 = fdiff * vec_W[b+l1];

//             vec_W[b+l1] -= temp1 * eta;
//             vec_W_cooccur[b+l2] -= temp2 * eta;
//         }

//         vec_b[record.w1] -= fdiff*eta ;
//         vec_b_cooccur[record.w2] -= fdiff*eta;

//     }

// }

void Glove::save_parameters(){
    std::ofstream out(vector_file);
    if(!out.is_open()){
        std::cerr << "error open word vector file!"<<std::endl;
        return;
    }

    for(int i=0;i<vocab_size;++i){
        out << hash_vocab[i];
        for(int j=0;j<vector_size;++j){
            out << "," << vec_W[i*vector_size+j]+vec_W_cooccur[i*vector_size+j];
        }
        out <<std::endl;
    }

    out.close();
}