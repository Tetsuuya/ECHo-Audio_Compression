#include <iostream>
#include <vector>
#include <map>
#include <queue>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <cstring>
#include <memory>
#include <algorithm>
#include <sndfile.h>
#include <bitset>

using namespace std;

const int CHUNK_SIZE = 4096;  // Reduced chunk size for better memory handling
const int QUANTIZATION_FACTOR = 64;
const double QUANTIZATION_THRESHOLD = 5.0;

struct AudioMetadata {
    int samplerate;
    int channels;
    sf_count_t frames;
};

vector<vector<double>> precomputeDCTMatrix(int N) {
    vector<vector<double>> dctMatrix(N, vector<double>(N));
    for (int k = 0; k < N; ++k) {
        double scale = (k == 0) ? sqrt(1.0/N) : sqrt(2.0/N);
        for (int n = 0; n < N; ++n) {
            dctMatrix[k][n] = scale * cos(M_PI * k * (2 * n + 1) / (2.0 * N));
        }
    }
    return dctMatrix;
}

vector<double> applyDCT(const vector<int16_t>& signal, int start, int length, const vector<vector<double>>& dctMatrix) {
    vector<double> transformed(length, 0.0);
    for (int k = 0; k < length; ++k) {
        for (int n = 0; n < length; ++n) {
            transformed[k] += signal[start + n] * dctMatrix[k][n];
        }
    }
    return transformed;
}

vector<int16_t> inverseDCT(const vector<double>& transformed, const vector<vector<double>>& dctMatrix) {
    int length = transformed.size();
    vector<int16_t> signal(length, 0);
    for (int n = 0; n < length; ++n) {
        double sum = 0.0;
        for (int k = 0; k < length; ++k) {
            sum += transformed[k] * dctMatrix[k][n];
        }
        signal[n] = static_cast<int16_t>(round(sum));
    }
    return signal;
}

vector<pair<int16_t, uint16_t>> runLengthEncode(const vector<int16_t>& data) {
    vector<pair<int16_t, uint16_t>> encoded;
    if (data.empty()) return encoded;
    
    int zero_run = 0;
    for (int16_t val : data) {
        if (val == 0) {
            zero_run++;
        } else {
            if (zero_run > 0) {
                encoded.emplace_back(0, zero_run);
                zero_run = 0;
            }
            encoded.emplace_back(val, 1);
        }
    }
    if (zero_run > 0) {
        encoded.emplace_back(0, zero_run);
    }
    
    return encoded;
}

vector<int16_t> runLengthDecode(const vector<pair<int16_t, uint16_t>>& encoded) {
    vector<int16_t> decoded;
    for (const auto& pair : encoded) {
        decoded.insert(decoded.end(), pair.second, pair.first);
    }
    return decoded;
}

struct HuffmanNode {
    int16_t symbol;
    int frequency;
    unique_ptr<HuffmanNode> left;
    unique_ptr<HuffmanNode> right;

    HuffmanNode(int16_t sym, int freq) : symbol(sym), frequency(freq) {}
};

struct CompareNode {
    bool operator()(const HuffmanNode* lhs, const HuffmanNode* rhs) const {
        return lhs->frequency > rhs->frequency;
    }
};

class HuffmanEncoder {
public:
    void buildTree(const vector<int16_t>& data) {
        map<int16_t, int> freqMap;
        for (int16_t symbol : data) {
            freqMap[symbol]++;
        }

        priority_queue<HuffmanNode*, vector<HuffmanNode*>, CompareNode> pq;
        for (const auto& pair : freqMap) {
            pq.push(new HuffmanNode(pair.first, pair.second));
        }

        while (pq.size() > 1) {
            auto left = unique_ptr<HuffmanNode>(pq.top());
            pq.pop();
            auto right = unique_ptr<HuffmanNode>(pq.top());
            pq.pop();

            auto newNode = new HuffmanNode(0, left->frequency + right->frequency);
            newNode->left = move(left);
            newNode->right = move(right);
            pq.push(newNode);
        }

        if (!pq.empty()) {
            root.reset(pq.top());
            generateCodes(root.get(), "");
        }
    }

    vector<bool> encode(const vector<int16_t>& data) {
        vector<bool> encodedData;
        for (int16_t symbol : data) {
            const string& code = codeMap.at(symbol);
            for (char bit : code) {
                encodedData.push_back(bit == '1');
            }
        }
        return encodedData;
    }

    vector<int16_t> decode(const vector<bool>& encodedData) {
        vector<int16_t> decodedData;
        HuffmanNode* current = root.get();
        
        for (bool bit : encodedData) {
            current = bit ? current->right.get() : current->left.get();
            if (!current->left && !current->right) {
                decodedData.push_back(current->symbol);
                current = root.get();
            }
        }
        
        return decodedData;
    }

    void saveCodebook(ofstream& outFile) {
        size_t codebookSize = codeMap.size();
        outFile.write(reinterpret_cast<const char*>(&codebookSize), sizeof(size_t));
        
        for (const auto& pair : codeMap) {
            outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(int16_t));
            uint8_t codeLength = pair.second.size();
            outFile.write(reinterpret_cast<const char*>(&codeLength), sizeof(uint8_t));
            
            uint8_t buffer = 0;
            int bitPos = 7;
            for (char bit : pair.second) {
                if (bit == '1') {
                    buffer |= (1 << bitPos);
                }
                bitPos--;
                if (bitPos < 0) {
                    outFile.write(reinterpret_cast<const char*>(&buffer), sizeof(uint8_t));
                    buffer = 0;
                    bitPos = 7;
                }
            }
            if (bitPos != 7) {
                outFile.write(reinterpret_cast<const char*>(&buffer), sizeof(uint8_t));
            }
        }
    }

    void loadCodebook(ifstream& inFile) {
        size_t codebookSize;
        inFile.read(reinterpret_cast<char*>(&codebookSize), sizeof(size_t));
        
        codeMap.clear();
        for (size_t i = 0; i < codebookSize; i++) {
            int16_t symbol;
            inFile.read(reinterpret_cast<char*>(&symbol), sizeof(int16_t));
            uint8_t codeLength;
            inFile.read(reinterpret_cast<char*>(&codeLength), sizeof(uint8_t));
            
            string code;
            uint8_t buffer;
            int bitsRead = 0;
            
            while (bitsRead < codeLength) {
                inFile.read(reinterpret_cast<char*>(&buffer), sizeof(uint8_t));
                for (int j = 7; j >= 0 && bitsRead < codeLength; j--, bitsRead++) {
                    code += (buffer & (1 << j)) ? '1' : '0';
                }
            }
            
            codeMap[symbol] = code;
        }
        
        root = make_unique<HuffmanNode>(0, 0);
        for (const auto& pair : codeMap) {
            HuffmanNode* current = root.get();
            for (char bit : pair.second) {
                if (bit == '0') {
                    if (!current->left) {
                        current->left = make_unique<HuffmanNode>(0, 0);
                    }
                    current = current->left.get();
                } else {
                    if (!current->right) {
                        current->right = make_unique<HuffmanNode>(0, 0);
                    }
                    current = current->right.get();
                }
            }
            current->symbol = pair.first;
        }
    }

private:
    unique_ptr<HuffmanNode> root;
    map<int16_t, string> codeMap;

    void generateCodes(HuffmanNode* node, const string& code) {
        if (!node) return;
        if (!node->left && !node->right) {
            codeMap[node->symbol] = code;
            return;
        }
        generateCodes(node->left.get(), code + "0");
        generateCodes(node->right.get(), code + "1");
    }
};

vector<uint8_t> packBits(const vector<bool>& bits) {
    vector<uint8_t> packed;
    uint8_t buffer = 0;
    int bitPos = 7;
    
    for (bool bit : bits) {
        if (bit) {
            buffer |= (1 << bitPos);
        }
        bitPos--;
        if (bitPos < 0) {
            packed.push_back(buffer);
            buffer = 0;
            bitPos = 7;
        }
    }
    
    if (bitPos != 7) {
        packed.push_back(buffer);
    }
    
    return packed;
}

vector<bool> unpackBits(const vector<uint8_t>& packed, size_t totalBits) {
    vector<bool> bits;
    bits.reserve(totalBits);
    
    for (uint8_t byte : packed) {
        for (int i = 7; i >= 0 && bits.size() < totalBits; i--) {
            bits.push_back((byte >> i) & 1);
        }
    }
    
    return bits;
}

void saveCompressedData(const vector<vector<pair<int16_t, uint16_t>>>& allRleData,
                      const vector<bool>& huffmanBits,
                      const AudioMetadata& metadata,
                      HuffmanEncoder& huffman,
                      const string& filename) {
    ofstream outFile(filename, ios::binary);
    if (!outFile) throw runtime_error("Failed to create output file");

    outFile.write(reinterpret_cast<const char*>(&metadata.samplerate), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&metadata.channels), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&metadata.frames), sizeof(sf_count_t));

    size_t numChunks = allRleData.size();
    outFile.write(reinterpret_cast<const char*>(&numChunks), sizeof(size_t));
    for (const auto& rleData : allRleData) {
        size_t numPairs = rleData.size();
        outFile.write(reinterpret_cast<const char*>(&numPairs), sizeof(size_t));
        for (const auto& pair : rleData) {
            outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(int16_t));
            outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(uint16_t));
        }
    }

    huffman.saveCodebook(outFile);

    vector<uint8_t> packedBits = packBits(huffmanBits);
    size_t numBits = huffmanBits.size();
    size_t numBytes = packedBits.size();
    outFile.write(reinterpret_cast<const char*>(&numBits), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(&numBytes), sizeof(size_t));
    outFile.write(reinterpret_cast<const char*>(packedBits.data()), numBytes);
}

void compressAudio(const string& filename) {
    SF_INFO sfinfo = {};
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) throw runtime_error("Error opening file: " + filename);

    cout << "Original audio: " << sfinfo.frames << " frames @ " 
         << sfinfo.samplerate << " Hz (" 
         << double(sfinfo.frames)/sfinfo.samplerate << " sec)" << endl;

    vector<int16_t> audioSamples(sfinfo.frames * sfinfo.channels);
    sf_read_short(file, audioSamples.data(), audioSamples.size());
    sf_close(file);

    vector<vector<pair<int16_t, uint16_t>>> allRleData;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);

    for (size_t i = 0; i < audioSamples.size(); i += CHUNK_SIZE) {
        size_t chunkSize = min(CHUNK_SIZE, static_cast<int>(audioSamples.size() - i));
        vector<double> dctTransformed = applyDCT(audioSamples, i, chunkSize, dctMatrix);
        
        vector<int16_t> quantized(chunkSize);
        for (size_t j = 0; j < chunkSize; j++) {
            double scale = (j < CHUNK_SIZE/4) ? 1.0 : (2.0 + (j - CHUNK_SIZE/4)/(CHUNK_SIZE/4));
            double val = dctTransformed[j] / (QUANTIZATION_FACTOR * scale);
            
            if (fabs(val) < QUANTIZATION_THRESHOLD * scale) {
                val = 0;
            }
            quantized[j] = static_cast<int16_t>(round(val));
        }
        
        allRleData.push_back(runLengthEncode(quantized));
    }

    vector<int16_t> allSymbols;
    for (const auto& chunk : allRleData) {
        for (const auto& pair : chunk) {
            allSymbols.push_back(pair.first);
        }
    }

    HuffmanEncoder huffman;
    huffman.buildTree(allSymbols);
    vector<bool> huffmanBits = huffman.encode(allSymbols);

    AudioMetadata metadata;
    metadata.samplerate = sfinfo.samplerate;
    metadata.channels = sfinfo.channels;
    metadata.frames = sfinfo.frames;

    saveCompressedData(allRleData, huffmanBits, metadata, huffman, "compressed.bin");
    
    ifstream inFile("compressed.bin", ios::binary | ios::ate);
    size_t compressedSize = inFile.tellg();
    inFile.close();
    
    double ratio = (double)compressedSize / (audioSamples.size() * sizeof(int16_t));
    cout << "Compression complete. Output: compressed.bin (" 
         << compressedSize / (1024.0 * 1024.0) << " MB)" << endl;
    cout << "Compression ratio: " << (ratio * 100) << "% of original size" << endl;
}

void decompressAudio(const string& outputFile) {
    ifstream inFile("compressed.bin", ios::binary);
    if (!inFile) throw runtime_error("Error opening compressed file");

    AudioMetadata metadata;
    inFile.read(reinterpret_cast<char*>(&metadata.samplerate), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&metadata.channels), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&metadata.frames), sizeof(sf_count_t));

    size_t numChunks;
    inFile.read(reinterpret_cast<char*>(&numChunks), sizeof(size_t));
    vector<vector<pair<int16_t, uint16_t>>> allRleData(numChunks);

    for (auto& rleData : allRleData) {
        size_t numPairs;
        inFile.read(reinterpret_cast<char*>(&numPairs), sizeof(size_t));
        rleData.resize(numPairs);
        for (auto& pair : rleData) {
            inFile.read(reinterpret_cast<char*>(&pair.first), sizeof(int16_t));
            inFile.read(reinterpret_cast<char*>(&pair.second), sizeof(uint16_t));
        }
    }

    HuffmanEncoder huffman;
    huffman.loadCodebook(inFile);

    size_t numBits, numBytes;
    inFile.read(reinterpret_cast<char*>(&numBits), sizeof(size_t));
    inFile.read(reinterpret_cast<char*>(&numBytes), sizeof(size_t));
    vector<uint8_t> packedBits(numBytes);
    inFile.read(reinterpret_cast<char*>(packedBits.data()), numBytes);
    inFile.close();

    vector<bool> huffmanBits = unpackBits(packedBits, numBits);
    vector<int16_t> decodedSymbols = huffman.decode(huffmanBits);

    size_t symbolIndex = 0;
    vector<vector<int16_t>> allDecodedData;
    for (const auto& rleData : allRleData) {
        vector<int16_t> chunkData;
        for (const auto& pair : rleData) {
            if (symbolIndex >= decodedSymbols.size()) {
                throw runtime_error("Not enough decoded symbols");
            }
            int16_t value = decodedSymbols[symbolIndex++];
            chunkData.insert(chunkData.end(), pair.second, value);
        }
        allDecodedData.push_back(chunkData);
    }

    vector<int16_t> audioSamples;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);
    for (const auto& chunk : allDecodedData) {
        vector<double> dctRestored(chunk.size());
        for (size_t i = 0; i < chunk.size(); i++) {
            double scale = (i < CHUNK_SIZE/4) ? 1.0 : (2.0 + (i - CHUNK_SIZE/4)/(CHUNK_SIZE/4));
            dctRestored[i] = chunk[i] * QUANTIZATION_FACTOR * scale;
        }
        
        vector<int16_t> signal = inverseDCT(dctRestored, dctMatrix);
        audioSamples.insert(audioSamples.end(), signal.begin(), signal.end());
    }

    size_t expectedSamples = metadata.frames * metadata.channels;
    if (audioSamples.size() > expectedSamples) {
        audioSamples.resize(expectedSamples);
    }

    cout << "Decompressed audio: " << audioSamples.size()/metadata.channels << " frames @ " 
         << metadata.samplerate << " Hz (" 
         << double(audioSamples.size())/metadata.samplerate/metadata.channels << " sec)" << endl;

    SF_INFO sfinfo = {};
    sfinfo.samplerate = metadata.samplerate;
    sfinfo.channels = metadata.channels;
    sfinfo.frames = audioSamples.size() / metadata.channels;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outFile = sf_open(outputFile.c_str(), SFM_WRITE, &sfinfo);
    if (!outFile) throw runtime_error("Error opening output file");
    sf_write_short(outFile, audioSamples.data(), audioSamples.size());
    sf_close(outFile);

    cout << "Decompression complete. Output: " << outputFile << endl;
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && strcmp(argv[1], "-d") == 0) {
            decompressAudio("output.wav");
        } else {
            if (argc < 2) {
                cerr << "Usage: " << argv[0] << " <input.wav> (for compression)" << endl;
                cerr << "       " << argv[0] << " -d (for decompression)" << endl;
                return 1;
            }
            compressAudio(argv[1]);
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}