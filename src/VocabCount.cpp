#include "VocabCount.h"

#include <unordered_map>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>

using namespace std;

VocabCount::VocabCount():trancate_frequency(1){

}

VocabCount::~VocabCount(){

}

bool compare_vocab_frequency(const VocabCount::VocabFrequency &a,const VocabCount::VocabFrequency &b){
    return a.frequency > b.frequency;
}

void VocabCount::generate_vocab(std::string file_name){
    ifstream in(file_name);
    if(!in.is_open())
        cerr << "error opening corpus！" << endl;

    std::string word;
    while(!in.eof()){
        in >> word;
        std::unordered_map<std::string,long>::iterator got = hash_vocab.find(word);
        if(got == hash_vocab.end()){
            std::pair<std::string,long> temp_pair(word,1);
            hash_vocab.insert(temp_pair);
        }
        else
            got->second++;
    }
        
    in.close();
}

void VocabCount::get_vocab_vector(){
    for(std::unordered_map<std::string,long>::const_iterator iter = hash_vocab.cbegin();iter != hash_vocab.cend();++iter){
        vec_vocab_frequency.push_back(VocabFrequency(iter->first,iter->second));
    }

    sort(vec_vocab_frequency.begin(),vec_vocab_frequency.end(),compare_vocab_frequency);
}

void VocabCount::trancate_vocab(long trancate_frequency = 1){
    set_trancate_frequency(trancate_frequency);
    for(std::vector<VocabFrequency>::iterator iter = vec_vocab_frequency.begin();iter!=vec_vocab_frequency.end();iter++){
        if(iter->frequency < trancate_frequency){
            vec_vocab_frequency.erase(iter,vec_vocab_frequency.end());
            break;
        }
    }
}

void VocabCount::save_to_file(std::string file_name){
    ofstream out(file_name);
    if(!out.is_open())  cerr << "error vocab file！" << endl;
    for(std::vector<VocabFrequency>::const_iterator iter = vec_vocab_frequency.cbegin();iter!=vec_vocab_frequency.cend();iter++){
        out << iter->word << " " << iter->frequency << endl;
    }
    out.close();
}
