#include <unordered_map>
#include <string>
#include <vector>

class VocabCount{
public:
    struct VocabFrequency{
        VocabFrequency(std::string word,long frequency):word(word),frequency(frequency){};
        std::string word;
        long frequency;
    };
    VocabCount();
    ~VocabCount();
    void generate_vocab(std::string file_name);
    void get_vocab_vector();
    void trancate_vocab(long trancate_frequency);
    void save_to_file(std::string file_name);
    inline void set_trancate_frequency(long trancate_frequency){this->trancate_frequency = trancate_frequency;};
    long get_trancate_frequency(){return this->trancate_frequency;};

private:
    long trancate_frequency;
    std::unordered_map<std::string, long> hash_vocab;
    std::vector<VocabFrequency> vec_vocab_frequency;
    
};