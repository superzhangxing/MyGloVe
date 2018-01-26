#include <iostream>
#include <unordered_map>
#include <string>
#include <fstream>

#include "VocabCount.h"
#include "Cooccurrence.h"
#include "Glove.h"

int main(){
    //test
    std::cout << "hello" <<std::endl;
    std::unordered_map<int,int> hash_a;
    hash_a[1] = 1;
    std::cout << hash_a[1] << std::endl;

    std::ofstream out("a.test",std::ofstream::binary);
    if(!out.is_open()){
        std::cerr<<"error open out test file!"<<std::endl;
        return 0;
    }
    for(int i=0;i<100;++i){
        out.write((const char*)&i,sizeof(int));
    }

    out.close();

    std::ifstream in("a.test",std::ifstream::binary);
    if(!in.is_open()){
        std::cerr<<"error open in test file!"<<std::endl;
        return 0;
    }
    int count1 = 0;
    int count2 = 0;
    int temp;
    // while(!in.eof()){
    //     in.read((char *)&temp,sizeof(int));
    //     count1++;
    // }
    // in.seekg(0,in.beg);
    while(in.peek()!=EOF){
        in.read((char *)&temp,sizeof(int));
        count2++;
    }

    in.close();
    std::cerr<<"count1: "<< count1<<std::endl;
    std::cerr<<"count2: "<< count2<<std::endl;


    // VocabCount * vocab_count = new VocabCount();
    // vocab_count->generate_vocab(std::string("/home/xing/Code/MyGloVe/text8"));
    // vocab_count->get_vocab_vector();
    // vocab_count->trancate_vocab(5);
    // vocab_count->save_to_file(std::string("/home/xing/Code/MyGloVe/vocab.txt"));
    // delete vocab_count;

    // Cooccurrence * cooccurrence_ptr = new Cooccurrence(15,4,1,"cooccur");
    // cooccurrence_ptr->generate_hash_vocab("/home/xing/Code/MyGloVe/vocab.txt");
    // cooccurrence_ptr->statistic_cooccurrence_record("/home/xing/Code/MyGloVe/text8");
    // cooccurrence_ptr->merge_file();
    // cooccurrence_ptr->shuffle();
    // delete cooccurrence_ptr;

    Glove * glove_ptr = new Glove();
    glove_ptr->train_glove();
    glove_ptr->save_parameters();



    return 1;
}