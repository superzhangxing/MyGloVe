#include "Cooccurrence.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <array>
#include <queue>
#include <algorithm>
#include <math.h>
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock

#include <assert.h>

using namespace std;

const double LIMIT = 0.0000001; 
const long long GIGA = 1024*1024*1024;

typedef std::pair<Cooccurrence::CooccurrenceRecord, int> record_pair;

// read word form file stream,skip '/n'
char temp_word[1000];
bool get_word(std::ifstream &in, std::string &word){
    int i = 0, ch;
    while (/*!in.eof()*/in.peek()!=EOF) {
        ch = in.get();
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (i > 0) {
                if (ch == '\n') in.unget();
                break;
            }
            if (ch == '\n') return false;
            else continue;
        }
        temp_word[i++] = ch;
        if (i >= 1000 - 1) i--;   // truncate words that exceed max length
    }
    temp_word[i] = 0;
    word = std::string(temp_word);
    return true;
}

bool cooccurrence_record_compare(const Cooccurrence::CooccurrenceRecord& record1,const Cooccurrence::CooccurrenceRecord& record2){
    if(record1.w1 == record2.w1) return (record1.w2 < record2.w2);
    else return (record1.w1 < record2.w1);
}

// bool cooccurrence_record_pair_compare(const record_pair& pair1, const record_pair& pair2){
//     return pair1.first.w1<pair2.first.w1 ? true:(pair1.first.w2<pair2.first.w2);
// }

// class cooccurrence_record_compare{
// public:
//     bool operator()(const Cooccurrence::CooccurrenceRecord& record1, const Cooccurrence::CooccurrenceRecord& record2){
//         return record1.w1<record2.w1 ? true:(record1.w2<record2.w2);
//     }
// }record_compare_object;

//priority default 
class cooccurrence_record_pair_compare{
public:
    bool operator()(const record_pair& pair1,const record_pair& pair2){
        if(pair1.first.w1 == pair2.first.w1) return (pair1.first.w2 > pair2.first.w2);
        else return (pair1.first.w1 > pair2.first.w1);
    }
};

Cooccurrence::Cooccurrence(unsigned int window_size ,unsigned int memory_limit, int symmetric, std::string record_file_head){
    this->window_size = window_size;
    this->memory_limit = memory_limit;
    this->symmetric = symmetric;
    this->record_file_head = record_file_head;

    this->shuffle_file_head = record_file_head + std::string("_shuffle");
    calculate_parameters();

}

Cooccurrence::~Cooccurrence(){

}

void Cooccurrence::generate_hash_vocab(std::string file_vocab){
    std::ifstream in(file_vocab,std::ifstream::in);
    if(!in.is_open()) std::cerr << "error open vocab file!" << std::endl;

    std::string word;
    int frequency;
    int rank = 0;

    // bool good = in.good();
    // bool eof = in.eof();
    // bool fail = in.fail();
    // bool bad = in.fail();

    while(in.good()){
        in >> word >> frequency;
        std::pair<std::string,int> temp_pair(word,rank);
        hash_vocab.insert(temp_pair);
        ++rank;
    }
    vocab_size = hash_vocab.size();

    in.close();
}

void Cooccurrence::statistic_cooccurrence_record(std::string file_corpus){
    //output information
    std::cerr<<"start scan corpus"<<std::endl;

    std::ifstream in_corpus(file_corpus);
    if(!in_corpus.is_open()) std::cerr << "error open corpus file!"<< std::endl;

    //store word rank in window
    std::vector<int> vec_history(window_size,0);
    //help to store the index of cooccurrence record pair stored in dense matrix directly
    std::vector<long long>vec_lookup(vocab_size,0);
    vec_lookup[0] = 0;
    for(long long i=1; i<=vocab_size;++i){
        if(vocab_size<max_product/(i)) 
            vec_lookup[i] = vec_lookup[i-1] + vocab_size;
        else
            vec_lookup[i] = vec_lookup[i-1] + max_product/(i);
    }
    //store all dense record weight
    std::vector<double> vec_matrix(vec_lookup[vocab_size],0.0);
    //store all sparse record 
    std::vector<CooccurrenceRecord> vec_sparse_record(overflow_length);

    std::string word1,word2;
    int w1,w2;
    int i = 0;
    int count = 0;
    int index_sparse_record = 0;
    int index_sparse_file = 1;
    while(/*!in_corpus.eof()*/in_corpus.peek()!=EOF){ // scan the corpus and statistic cooccurrence record
        //output process information
        if(count%100000 == 0){
            std::string process_info = std::string("\033[00Gscan ") + std::to_string(count) + std::string(" tokens in corpus!");
            std::cerr << process_info;
        }
        //in_corpus >> word1;
        bool flag = get_word(in_corpus,word1);
        if(!flag){
            i=0;
            continue;   //new line
        }
        
        ++count;
        
        std::unordered_map<std::string,int>::const_iterator got = hash_vocab.find(word1);
        if(got == hash_vocab.end())
            continue;  // skip word not in vocab
        else
            w1 = got->second;

        for(int j= (i - window_size>0)?(i-window_size):0;j<i;++j){
            w2 = vec_history[j%window_size];
            double weight = 1.0/(double)(i-j);
            if(w1 <= max_product/(w2+1)-1){// store in dense matrix
                vec_matrix[vec_lookup[w1]+w2] += weight;
                if(symmetric == 1) vec_matrix[vec_lookup[w2]+w1] += weight;
            }
            else{ // store in sparse matrix
                vec_sparse_record[index_sparse_record].w1 = w1;
                vec_sparse_record[index_sparse_record].w2 = w2;
                vec_sparse_record[index_sparse_record].weight = weight;
                ++index_sparse_record;
                if(symmetric == 1){
                    vec_sparse_record[index_sparse_record].w1 = w2;
                    vec_sparse_record[index_sparse_record].w2 = w1;
                    vec_sparse_record[index_sparse_record].weight = weight;
                    ++index_sparse_record;    
                }

                // sparse matrix is full
                if(index_sparse_record == overflow_length){
                    std::string sparse_file_name = record_file_head + std::to_string(index_sparse_file);
                    save_sparse_record_to_file(vec_sparse_record,index_sparse_record,sparse_file_name);
                    index_sparse_record = 0;
                    ++index_sparse_file;
                }
            }
        }

        //update window
        vec_history[i%window_size] = w1; 
        ++i;
    }
    //output scan information
    std::string process_info = std::string("\033[00Gscan ") + std::to_string(count) + std::string(" tokens in corpus!");
    std::cerr << process_info << std::endl; 

    //save dense record to file
    std::string dense_file_name = record_file_head + std::to_string(0);
    save_dense_record_to_file(vec_lookup,vec_matrix,dense_file_name);
    
    //save remain sparse record to file
    if(index_sparse_record==0){
        index_sparse_file--;
    }
    else{
        std::string sparse_file_name = record_file_head + std::to_string(index_sparse_file);
        save_sparse_record_to_file(vec_sparse_record,index_sparse_record,sparse_file_name);
    }

    record_file_number = index_sparse_file+1;
    
    
}

bool Cooccurrence::save_dense_record_to_file(const std::vector<long long>& vec_lookup,const std::vector<double>& vec_matrix,std::string file_record){
    std::ofstream out(file_record,std::ios::binary);
    if(!out.is_open()){
        std::cerr << "error open dense record file!"<<std::endl;
        return false;
    }

    int i=0,w1=0,w2=0;
    int len = vec_lookup.size();
    for(w1=0;w1<vocab_size;++w1){
        for(w2=0;w2<vec_lookup[w1+1]-vec_lookup[w1];++w2){
            double weight = vec_matrix[vec_lookup[w1]+w2];
            if(weight > LIMIT){// store non-zero elements
                CooccurrenceRecord record = {w1,w2,weight};
                int a = sizeof(record);
                int b = sizeof(CooccurrenceRecord);
                out.write((const char*)&record, sizeof(record));
            }
        }
    }

    out.close();
    return true;
}

bool Cooccurrence::save_sparse_record_to_file(std::vector<CooccurrenceRecord>& vec_spare_record,size_t size,std::string file_record){
    assert(size <= vec_spare_record.size());

    std::ofstream out(file_record,std::ios::binary);
    if(!out.is_open()){
        std::cerr << "error open sparse record file!" << std::endl;
        return false;
    }

    // sort records by its rank
    sort(vec_spare_record.begin(),vec_spare_record.begin()+size,cooccurrence_record_compare);

    // merge the same cooccurrence records and save them to file
    CooccurrenceRecord record,old_record;
    record = vec_spare_record[0];
    old_record = vec_spare_record[0];
    for(int i=1;i<size;++i){
        if(vec_spare_record[i].w1 == record.w1 && vec_spare_record[i].w2 == record.w2){
            record.weight += vec_spare_record[i].weight;
        }
        else{
            out.write((const char*) &record, sizeof(record));
            record = vec_spare_record[i];
        }
    }
    out.write((const char*) &record, sizeof(record));

    out.close();
    return true;
}

bool Cooccurrence::merge_file(){
    // output merge file information
    std::cerr << "start merge file"<<std::endl;

    std::priority_queue<record_pair,std::vector<record_pair>,cooccurrence_record_pair_compare> record_pair_priority_queue;

    assert(record_file_number > 0);

    // file stream vector
    std::vector<std::ifstream> vec_in(record_file_number);
    for(int i=0;i<record_file_number;++i){
        std::string record_file_name = record_file_head + std::to_string(i);
        vec_in[i].open(record_file_name, std::ifstream::binary|std::ifstream::in);
        if(!vec_in[i].is_open()) std::cerr << "error open temp record file!" << std::endl;
    }

    std::ofstream out(record_file_head, std::ofstream::binary|std::ofstream::out);
    if(!out.is_open()) std::cerr << "error open record file!" << std::endl;

    CooccurrenceRecord record,old_record;
    vec_in[0].read((char *) &record, sizeof(record));
    old_record = record;
    for(int i=0; i< record_file_number;++i){
        if(/*!vec_in[i].eof()*/vec_in[i].peek()!=EOF){
            vec_in[i].read((char *) &record, sizeof(record));
            record_pair pair(record, i);
            record_pair_priority_queue.push(pair);
        }
    }

    // use prioroty queue to merge all records
    long long count = 0; 
    while(!record_pair_priority_queue.empty()){
        record_pair pair = record_pair_priority_queue.top();
        record_pair_priority_queue.pop();

        if(old_record.w1 == pair.first.w1 && old_record.w2 == pair.first.w2){
            old_record.weight += pair.first.weight;
        }
        else{
            if(old_record.weight < LIMIT){
                std::cerr<<"hehe";
            }
            out.write((const char*)&old_record, sizeof(old_record));
            old_record = pair.first;

            // output merge information
            count++;
            if(count%100000==0){
                std::string merge_info = std::string("\033[00Gmerge ")+std::to_string(count)+std::string(" coccurrence records");
                std::cerr << merge_info;
            }
        }

        int file_id = pair.second;
        // insert new record pair to priority queue, current file id is first
        for(int i=file_id;i<file_id+record_file_number; ++i){
            int insert_file_id = i%record_file_number;
            if(/*!vec_in[insert_file_id].eof()*/vec_in[insert_file_id].peek()!=EOF){
                vec_in[insert_file_id].read((char *)&record, sizeof(record));
                record_pair temp_pair(record,insert_file_id);
                record_pair_priority_queue.push(temp_pair);
                break;
            }
        }
    }
    if(old_record.weight < LIMIT){
        std::cerr<<"hehe";
    }
    out.write((const char*)&old_record, sizeof(old_record));
    ++count;
    std::string merge_info = std::string("\033[00Gmerge ")+std::to_string(count)+std::string(" coccurrence records");
    std::cerr << merge_info <<std::endl;

    out.close();

    return true;
}

void Cooccurrence::shuffle(){
    generate_temp_shuffle_file();
    merge_temp_shuffle_file();
}

bool Cooccurrence::generate_temp_shuffle_file(){
    std::ifstream in_cooccur(record_file_head, std::ifstream::binary);
    if(!in_cooccur.is_open()){
        std::cerr << "error open cooccur file!"<< std::endl;
        return false;
    }

    in_cooccur.seekg(0,in_cooccur.end);
    int length = in_cooccur.tellg();
    in_cooccur.seekg(0,in_cooccur.beg);

    std::vector<CooccurrenceRecord> vec_record(shuffle_array_size);
    CooccurrenceRecord * vec_ptr = vec_record.data();
    bool file_eof = false;
    int shuffle_file_id = 0;
    while(true){
        for(int i=0; i<shuffle_array_size; ++i){
            if(/*!in_cooccur.eof()*/in_cooccur.peek()!=EOF){
                //in_cooccur.read((char *)&vec_record[i],sizeof(CooccurrenceRecord));
                in_cooccur.read((char *)(vec_ptr+i),sizeof(CooccurrenceRecord));
                if((vec_ptr+i)->weight<LIMIT){
                    std::cerr<<"wrong record"<<std::endl;
                }
            }
            else{// the last chunk
                file_eof = true;
                if(i!=0){
                    for(int j=0;j<i;++j){
                        if(vec_record[j].weight<LIMIT){
                            std::cerr<<"wrong record"<<std::endl;
                        }
                    }
                    std::random_shuffle(vec_record.begin(),vec_record.begin()+i);
                    for(int j=0;j<i;++j){
                        if(vec_record[j].weight<LIMIT){
                            std::cerr<<"wrong record"<<std::endl;
                        }
                    }
                    std::string temp_file_name = shuffle_file_head+std::to_string(shuffle_file_id++);
                    save_temp_shuffle_file(vec_record,i,temp_file_name);
                    std::cerr<<std::string("\033[00Gprocessed temp shuffle file: ")+std::to_string(shuffle_file_id-1);
                }
                break;
            }
        }
        if(file_eof) break;

        // the normal chunk
        std::random_shuffle(vec_record.begin(),vec_record.end());
        std::string temp_file_name = shuffle_file_head+std::to_string(shuffle_file_id++);
        save_temp_shuffle_file(vec_record,shuffle_array_size,temp_file_name);
        
        std::cerr<<std::string("\033[00Gprocessed temp shuffle file: ")+std::to_string(shuffle_file_id-1);

    }
    shuffle_file_number = shuffle_file_id;
    std::cerr<<std::endl;

    in_cooccur.close();

    return true;
}

bool Cooccurrence::save_temp_shuffle_file(const std::vector<CooccurrenceRecord>& vec_record,int size,std::string file_name){
    assert(size >  0);
    assert(size <= vec_record.size());

    std::ofstream out(file_name, std::ofstream::binary);
    if(!out.is_open()){
        std::cerr << std::string("error open shuffle file: ")+file_name+std::string(" !")<<std::endl;
        return false;
    }

    const CooccurrenceRecord * vec_ptr = vec_record.data();
    for(int i=0;i<size;++i){
        out.write((const char *)(vec_ptr+i),sizeof(CooccurrenceRecord));
    }

    out.close();
    return true;
}

bool Cooccurrence::merge_temp_shuffle_file(){
    if(shuffle_file_number == 0){
        std::cerr<<std::string("no merge file!")<<std::endl;
        return true;
    }

    std::ofstream out_merge(shuffle_file_head,std::ofstream::binary);
    if(!out_merge.is_open()){
        std::cerr<<std::string("error open merge shuffle file!")<<std::endl;
        return false;
    }

    std::vector<std::ifstream> vec_in_temp(shuffle_file_number);
    for(int i=0;i<shuffle_file_number;++i){
        std::string temp_file_name = shuffle_file_head+std::to_string(i);
        vec_in_temp[i].open(temp_file_name,std::ifstream::binary);
        if(!vec_in_temp[i].is_open()){
            std::cerr<<std::string("error open temp shuffle file or merge!")<<std::endl;
            return false;
        }
    }

    std::vector<CooccurrenceRecord> vec_record(shuffle_array_size);
    CooccurrenceRecord * read_vec_ptr = vec_record.data();
    while(true){
        int k=0;
        for(int i=0;i<shuffle_file_number;++i){
            if(/*!vec_in_temp[i].eof()*/vec_in_temp[i].peek()!=EOF){
                for(int j=0; j<shuffle_array_size/shuffle_file_number;++j){
                    if(/*!vec_in_temp[i].eof()*/vec_in_temp[i].peek()!=EOF){
                        //vec_in_temp[i].read((char *)&vec_record[k],sizeof(CooccurrenceRecord));
                        vec_in_temp[i].read((char *)(read_vec_ptr+k),sizeof(CooccurrenceRecord));
                        k++;
                    }
                    else
                        break;
                }
            }
        }

        if(k==0) break;//all temp file has been merged

        std::random_shuffle(vec_record.begin(),vec_record.begin()+k);
        CooccurrenceRecord * vec_ptr = vec_record.data();
        for(int i=0;i<k;++i){
            out_merge.write((const char *)(vec_ptr+i),sizeof(CooccurrenceRecord));
        }
        
    }

    out_merge.close();
    return true;
}

void Cooccurrence::calculate_parameters(){
    double rlimit = 0.85 * (double)memory_limit * GIGA / sizeof(CooccurrenceRecord);
    long long n=2;
    while(n*(log(n+1)+0.5)<rlimit){// 1+1/2+1/3+...1/n~ log(n+1)+r;r=0.5772156649
        ++n;
    }
    // while(n * (log(n) + 0.1544313298)<rlimit){
    //     ++n;
    // }
    max_product = n;
    overflow_length = rlimit/6;
    overflow_length = (overflow_length/2)*2;

    shuffle_array_size = 0.95 * (double)memory_limit * GIGA / sizeof(CooccurrenceRecord);
}