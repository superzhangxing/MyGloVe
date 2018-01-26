#include <vector>
#include <string>
#include <unordered_map>

class Cooccurrence{
public:
    struct CooccurrenceRecord{
        int w1,w2;
        double weight;
        //CooccurrenceRecord():w1(0),w2(0),weight(0.0){};
    };

    Cooccurrence(unsigned int window_size, unsigned int memory_limit , int symmetric, std::string record_file_head);
    ~Cooccurrence();

    void generate_hash_vocab(std::string file_vocab);
    void statistic_cooccurrence_record(std::string file_corpus);
    bool save_dense_record_to_file(const std::vector<long long>& vec_lookup,const std::vector<double>& vec_matrix,std::string file_record);
    bool save_sparse_record_to_file(std::vector<CooccurrenceRecord>& vec_spare_record,size_t size,std::string file_record);
    bool merge_file();

    void shuffle();
    bool generate_temp_shuffle_file();
    bool save_temp_shuffle_file(const std::vector<CooccurrenceRecord>& vec_record,int size,std::string file_name);
    bool merge_temp_shuffle_file();

    inline void set_window_size(unsigned int window_size){this->window_size = window_size;};
    inline unsigned int get_window_size(){return this->window_size;};
    inline void set_memory_limit(unsigned int memory_limit){this->memory_limit = memory_limit;};
    inline unsigned int get_memory_limit(){return this->memory_limit;};
    inline void set_symmetric(int symmetric){this->symmetric = symmetric;};
    inline int get_symmetric(){return symmetric;};
private:
    void calculate_parameters();
    unsigned int window_size;
    int vocab_size;
    unsigned int memory_limit;
    long long max_product;
    long long overflow_length;
    int symmetric;                 // 0 left,1 left-right,
    int record_file_number;        
    std::string record_file_head;

    long long shuffle_array_size;
    std::string shuffle_file_head;
    unsigned int shuffle_file_number;  // temp shuffle file number

    //a hash map for word to rank
    std::unordered_map<std::string,int> hash_vocab;
};