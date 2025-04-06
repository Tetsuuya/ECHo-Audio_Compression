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

using namespace std;

const int CHUNK_SIZE = 1024;
const int QUANTIZATION_FACTOR = 1;

// Audio metadata storage
struct AudioMetadata {
    int samplerate;
    int channels;
    sf_count_t frames;
};

// --- DCT Functions ---
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

// --- Run-Length Encoding ---
vector<pair<int, int>> runLengthEncode(const vector<int>& data) {
    vector<pair<int, int>> encoded;
    if (data.empty()) return encoded;
    
    int prev = data[0];
    int count = 1;
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] == prev) {
            count++;
        } else {
            encoded.emplace_back(prev, count);
            prev = data[i];
            count = 1;
        }
    }
    encoded.emplace_back(prev, count);
    return encoded;
}

vector<int> runLengthDecode(const vector<pair<int, int>>& encoded) {
    vector<int> decoded;
    for (const auto& pair : encoded) {
        decoded.insert(decoded.end(), pair.second, pair.first);
    }
    return decoded;
}

// --- Huffman Encoding ---
struct HuffmanNode {
    int symbol;
    int frequency;
    unique_ptr<HuffmanNode> left;
    unique_ptr<HuffmanNode> right;

    HuffmanNode(int sym, int freq) : symbol(sym), frequency(freq) {}
};

struct CompareNode {
    bool operator()(const HuffmanNode* lhs, const HuffmanNode* rhs) const {
        return lhs->frequency > rhs->frequency;
    }
};

class HuffmanEncoder {
public:
    void buildTree(const vector<int>& data) {
        map<int, int> freqMap;
        for (int symbol : data) {
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

            auto newNode = new HuffmanNode(-1, left->frequency + right->frequency);
            newNode->left = move(left);
            newNode->right = move(right);
            pq.push(newNode);
        }

        if (!pq.empty()) {
            root.reset(pq.top());
            generateCodes(root.get(), "");
        }
    }

    vector<string> encode(const vector<int>& data) {
        vector<string> encodedData;
        for (int symbol : data) {
            encodedData.push_back(codeMap.at(symbol));
        }
        return encodedData;
    }

    vector<int> decode(const vector<string>& encodedData) {
        vector<int> decodedData;
        for (const string& code : encodedData) {
            HuffmanNode* current = root.get();
            for (char bit : code) {
                current = (bit == '0') ? current->left.get() : current->right.get();
                if (current == nullptr) {
                    throw runtime_error("Invalid Huffman code");
                }
            }
            decodedData.push_back(current->symbol);
        }
        return decodedData;
    }

private:
    unique_ptr<HuffmanNode> root;
    map<int, string> codeMap;

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

// --- Compression ---
void saveCompressedData(const vector<vector<pair<int, int>>>& allRleData,
                      const vector<string>& huffmanEncodedData,
                      const AudioMetadata& metadata,
                      const string& filename) {
    ofstream outFile(filename, ios::binary);
    if (!outFile) throw runtime_error("Failed to create output file");

    // Save metadata
    outFile.write(reinterpret_cast<const char*>(&metadata.samplerate), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&metadata.channels), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&metadata.frames), sizeof(sf_count_t));

    // Save RLE data
    size_t numChunks = allRleData.size();
    outFile.write(reinterpret_cast<const char*>(&numChunks), sizeof(size_t));
    for (const auto& rleData : allRleData) {
        size_t numPairs = rleData.size();
        outFile.write(reinterpret_cast<const char*>(&numPairs), sizeof(size_t));
        for (const auto& pair : rleData) {
            outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(int));
            outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(int));
        }
    }

    // Save Huffman data
    size_t numEncodedSymbols = huffmanEncodedData.size();
    outFile.write(reinterpret_cast<const char*>(&numEncodedSymbols), sizeof(size_t));
    for (const string& code : huffmanEncodedData) {
        uint8_t length = code.size();
        outFile.write(reinterpret_cast<const char*>(&length), sizeof(uint8_t));
        outFile.write(code.c_str(), length);
    }
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

    vector<vector<pair<int, int>>> allRleData;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);

    for (size_t i = 0; i < audioSamples.size(); i += CHUNK_SIZE) {
        size_t chunkSize = min(CHUNK_SIZE, static_cast<int>(audioSamples.size() - i));
        vector<double> dctTransformed = applyDCT(audioSamples, i, chunkSize, dctMatrix);
        
        vector<int> quantized(chunkSize);
        transform(dctTransformed.begin(), dctTransformed.end(), quantized.begin(),
                 [](double val) { return static_cast<int>(round(val / QUANTIZATION_FACTOR)); });
        
        allRleData.push_back(runLengthEncode(quantized));
    }

    vector<int> allSymbols;
    for (const auto& chunk : allRleData) {
        for (const auto& pair : chunk) {
            allSymbols.push_back(pair.first);
        }
    }

    HuffmanEncoder huffman;
    huffman.buildTree(allSymbols);
    vector<string> huffmanEncodedData = huffman.encode(allSymbols);

    AudioMetadata metadata;
    metadata.samplerate = sfinfo.samplerate;
    metadata.channels = sfinfo.channels;
    metadata.frames = sfinfo.frames;

    saveCompressedData(allRleData, huffmanEncodedData, metadata, "compressed.bin");
    cout << "Compression complete. Output: compressed.bin" << endl;
}

// --- Decompression ---
void decompressAudio(const string& outputFile) {
    ifstream inFile("compressed.bin", ios::binary);
    if (!inFile) throw runtime_error("Error opening compressed file");

    // Read metadata
    AudioMetadata metadata;
    inFile.read(reinterpret_cast<char*>(&metadata.samplerate), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&metadata.channels), sizeof(int));
    inFile.read(reinterpret_cast<char*>(&metadata.frames), sizeof(sf_count_t));

    // Read RLE data
    size_t numChunks;
    inFile.read(reinterpret_cast<char*>(&numChunks), sizeof(size_t));
    vector<vector<pair<int, int>>> allRleData(numChunks);

    for (auto& rleData : allRleData) {
        size_t numPairs;
        inFile.read(reinterpret_cast<char*>(&numPairs), sizeof(size_t));
        rleData.resize(numPairs);
        for (auto& pair : rleData) {
            inFile.read(reinterpret_cast<char*>(&pair.first), sizeof(int));
            inFile.read(reinterpret_cast<char*>(&pair.second), sizeof(int));
        }
    }

    // Read Huffman data
    size_t numEncodedSymbols;
    inFile.read(reinterpret_cast<char*>(&numEncodedSymbols), sizeof(size_t));
    vector<string> huffmanEncodedData(numEncodedSymbols);
    
    for (string& code : huffmanEncodedData) {
        uint8_t length;
        inFile.read(reinterpret_cast<char*>(&length), sizeof(uint8_t));
        code.resize(length);
        inFile.read(&code[0], length);
    }
    inFile.close();

    // Decode
    HuffmanEncoder huffman;
    vector<int> allSymbols;
    for (const auto& chunk : allRleData) {
        for (const auto& pair : chunk) {
            allSymbols.push_back(pair.first);
        }
    }
    huffman.buildTree(allSymbols);
    
    vector<int> decodedSymbols = huffman.decode(huffmanEncodedData);

    // Reconstruct
    vector<vector<int>> allDecodedData;
    size_t symbolIndex = 0;
    for (const auto& rleData : allRleData) {
        vector<int> chunkData;
        for (const auto& pair : rleData) {
            if (symbolIndex >= decodedSymbols.size()) {
                throw runtime_error("Not enough decoded symbols");
            }
            int value = decodedSymbols[symbolIndex++];
            chunkData.insert(chunkData.end(), pair.second, value);
        }
        allDecodedData.push_back(chunkData);
    }

    // Inverse DCT
    vector<int16_t> audioSamples;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);
    for (const auto& chunk : allDecodedData) {
        vector<double> dctRestored(chunk.size());
        transform(chunk.begin(), chunk.end(), dctRestored.begin(),
                 [](int val) { return val * QUANTIZATION_FACTOR; });
        
        vector<int16_t> signal = inverseDCT(dctRestored, dctMatrix);
        audioSamples.insert(audioSamples.end(), signal.begin(), signal.end());
    }

    // Trim to original length if needed
    size_t expectedSamples = metadata.frames * metadata.channels;
    if (audioSamples.size() > expectedSamples) {
        audioSamples.resize(expectedSamples);
    }

    cout << "Decompressed audio: " << audioSamples.size()/metadata.channels << " frames @ " 
         << metadata.samplerate << " Hz (" 
         << double(audioSamples.size())/metadata.samplerate/metadata.channels << " sec)" << endl;

    // Save
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

// --- Main ---
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
