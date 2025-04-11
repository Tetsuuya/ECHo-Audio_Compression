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

const int CHUNK_SIZE = 4096;  // Keep chunk size reasonably large for better DCT
// Increase quantization to get better compression
const int QUANTIZATION_FACTOR = 256; // Increased from 64
const double QUANTIZATION_THRESHOLD = 0.7; // Lowered from 5.0

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

// Improved run-length encoding that handles small values more efficiently
vector<pair<int16_t, uint16_t>> runLengthEncode(const vector<int16_t>& data) {
    vector<pair<int16_t, uint16_t>> encoded;
    if (data.empty()) return encoded;
    
    int zero_run = 0;
    int16_t prev_val = 0;
    uint16_t run_count = 0;
    
    for (size_t i = 0; i < data.size(); i++) {
        int16_t val = data[i];
        
        if (val == 0) {
            // Handle zero runs specially
            if (run_count > 0) {
                encoded.emplace_back(prev_val, run_count);
                run_count = 0;
            }
            zero_run++;
        } else {
            if (zero_run > 0) {
                encoded.emplace_back(0, zero_run);
                zero_run = 0;
            }
            
            if (val == prev_val && run_count > 0) {
                run_count++;
            } else {
                if (run_count > 0) {
                    encoded.emplace_back(prev_val, run_count);
                }
                prev_val = val;
                run_count = 1;
            }
        }
    }
    
    // Handle the last run
    if (zero_run > 0) {
        encoded.emplace_back(0, zero_run);
    } else if (run_count > 0) {
        encoded.emplace_back(prev_val, run_count);
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

// Optimized Adaptive Huffman implementation
struct AdaptiveHuffmanNode {
    int16_t symbol;
    int weight;
    int order;
    AdaptiveHuffmanNode *parent, *left, *right;

    AdaptiveHuffmanNode(int16_t sym, int wt, int ord) 
        : symbol(sym), weight(wt), order(ord), parent(nullptr), left(nullptr), right(nullptr) {}
};

class AdaptiveHuffman {
private:
    map<int16_t, AdaptiveHuffmanNode*> symbolMap;
    AdaptiveHuffmanNode* root;
    int nextOrder;
    const int16_t NYT_SYMBOL = numeric_limits<int16_t>::max();

public:
    AdaptiveHuffman() {
        root = new AdaptiveHuffmanNode(NYT_SYMBOL, 0, 512);
        symbolMap[NYT_SYMBOL] = root;
        nextOrder = 511;
    }

    ~AdaptiveHuffman() {
        freeTree(root);
    }

    vector<bool> encode(int16_t symbol) {
        vector<bool> code;
        if (symbolMap.find(symbol) != symbolMap.end()) {
            AdaptiveHuffmanNode* node = symbolMap[symbol];
            code = getCode(node);
        } else {
            AdaptiveHuffmanNode* nytNode = symbolMap[NYT_SYMBOL];
            vector<bool> nytCode = getCode(nytNode);
            code.insert(code.end(), nytCode.begin(), nytCode.end());
            
            // Use adaptive bit width for symbols
            // Only use as many bits as needed to represent the symbol
            uint16_t absSymbol = abs(symbol);
            int bitWidth = 0;
            while ((1 << bitWidth) <= absSymbol) {
                bitWidth++;
            }
            
            // Write the bit width (4 bits can represent 0-15)
            for (int i = 3; i >= 0; i--) {
                code.push_back((bitWidth >> i) & 1);
            }
            
            // Write the sign bit (1 for negative, 0 for positive)
            code.push_back(symbol < 0);
            
            // Write the magnitude bits
            for (int i = bitWidth - 1; i >= 0; i--) {
                code.push_back((absSymbol >> i) & 1);
            }
        }
        updateTree(symbol);
        return code;
    }

    int16_t decode(const vector<bool>& code, size_t& pos) {
        AdaptiveHuffmanNode* node = root;
        int16_t symbol;
        
        // Traverse tree until we hit a leaf
        while (node->left || node->right) {
            if (pos >= code.size()) throw runtime_error("Invalid code sequence");
            bool bit = code[pos++];
            node = bit ? node->right : node->left;
        }
        
        if (node->symbol == NYT_SYMBOL) {
            // Read the bit width first (4 bits)
            if (pos + 4 > code.size()) throw runtime_error("Invalid code sequence");
            int bitWidth = 0;
            for (int i = 0; i < 4; i++) {
                bitWidth = (bitWidth << 1) | (code[pos++] ? 1 : 0);
            }
            
            // Read sign bit
            if (pos >= code.size()) throw runtime_error("Invalid code sequence");
            bool isNegative = code[pos++];
            
            // Read the magnitude bits
            if (pos + bitWidth > code.size()) throw runtime_error("Invalid code sequence");
            uint16_t magnitude = 0;
            for (int i = 0; i < bitWidth; i++) {
                magnitude = (magnitude << 1) | (code[pos++] ? 1 : 0);
            }
            
            symbol = isNegative ? -magnitude : magnitude;
        } else {
            symbol = node->symbol;
        }
        
        updateTree(symbol);
        return symbol;
    }

private:
    vector<bool> getCode(AdaptiveHuffmanNode* node) {
        vector<bool> code;
        while (node->parent) {
            code.push_back(node == node->parent->right);
            node = node->parent;
        }
        reverse(code.begin(), code.end());
        return code;
    }

    void updateTree(int16_t symbol) {
        AdaptiveHuffmanNode* node;
        
        if (symbolMap.find(symbol) != symbolMap.end()) {
            node = symbolMap[symbol];
        } else {
            AdaptiveHuffmanNode* nytNode = symbolMap[NYT_SYMBOL];
            
            AdaptiveHuffmanNode* internal = new AdaptiveHuffmanNode(0, 0, nextOrder--);
            internal->parent = nytNode->parent;
            if (nytNode->parent) {
                if (nytNode->parent->left == nytNode) {
                    nytNode->parent->left = internal;
                } else {
                    nytNode->parent->right = internal;
                }
            } else {
                root = internal;
            }
            
            AdaptiveHuffmanNode* newNYT = new AdaptiveHuffmanNode(NYT_SYMBOL, 0, nextOrder--);
            newNYT->parent = internal;
            internal->left = newNYT;
            symbolMap[NYT_SYMBOL] = newNYT;
            
            node = new AdaptiveHuffmanNode(symbol, 0, nextOrder--);
            node->parent = internal;
            internal->right = node;
            symbolMap[symbol] = node;
        }
        
        while (node) {
            AdaptiveHuffmanNode* toSwap = findNodeToSwap(node);
            if (toSwap && node != toSwap && node->parent != toSwap) {
                swapNodes(node, toSwap);
            }
            node->weight++;
            node = node->parent;
        }
    }

    AdaptiveHuffmanNode* findNodeToSwap(AdaptiveHuffmanNode* node) {
        AdaptiveHuffmanNode* candidate = nullptr;
        queue<AdaptiveHuffmanNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            AdaptiveHuffmanNode* current = q.front();
            q.pop();
            
            if (current != node && current->weight == node->weight && 
                current->order > node->order) {
                if (!candidate || current->order > candidate->order) {
                    candidate = current;
                }
            }
            
            if (current->left) q.push(current->left);
            if (current->right) q.push(current->right);
        }
        
        return candidate;
    }

    void swapNodes(AdaptiveHuffmanNode* a, AdaptiveHuffmanNode* b) {
        if (!a->parent || !b->parent) return;
        
        if (a->parent->left == a) {
            a->parent->left = b;
        } else {
            a->parent->right = b;
        }
        
        if (b->parent->left == b) {
            b->parent->left = a;
        } else {
            b->parent->right = a;
        }
        
        swap(a->parent, b->parent);
        swap(a->order, b->order);
    }

    void freeTree(AdaptiveHuffmanNode* node) {
        if (!node) return;
        freeTree(node->left);
        freeTree(node->right);
        delete node;
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

// Optimized compressed data format - minimize overhead
void saveCompressedData(const vector<vector<pair<int16_t, uint16_t>>>& allRleData,
                      const vector<bool>& adaptiveHuffmanBits,
                      const AudioMetadata& metadata,
                      const string& filename) {
    ofstream outFile(filename, ios::binary);
    if (!outFile) throw runtime_error("Failed to create output file");

    // Write header with minimum space
    uint32_t headerData[3] = {
        static_cast<uint32_t>(metadata.samplerate),
        static_cast<uint32_t>(metadata.channels),
        static_cast<uint32_t>(metadata.frames)
    };
    outFile.write(reinterpret_cast<const char*>(headerData), sizeof(headerData));

    // Write RLE data with optimized format
    uint32_t numChunks = allRleData.size();
    outFile.write(reinterpret_cast<const char*>(&numChunks), sizeof(uint32_t));
    
    for (const auto& rleData : allRleData) {
        uint32_t numPairs = rleData.size();
        outFile.write(reinterpret_cast<const char*>(&numPairs), sizeof(uint32_t));
        
        // Pack run length and value together where possible
        for (const auto& pair : rleData) {
            outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(int16_t));
            outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(uint16_t));
        }
    }

    // Write Huffman-encoded data
    vector<uint8_t> packedBits = packBits(adaptiveHuffmanBits);
    uint32_t numBits = adaptiveHuffmanBits.size();
    uint32_t numBytes = packedBits.size();
    outFile.write(reinterpret_cast<const char*>(&numBits), sizeof(uint32_t));
    outFile.write(reinterpret_cast<const char*>(&numBytes), sizeof(uint32_t));
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

    // Calculate original file size
    size_t originalSize = audioSamples.size() * sizeof(int16_t);
    cout << "Original size: " << (originalSize / 1024.0) << " KB" << endl;

    vector<vector<pair<int16_t, uint16_t>>> allRleData;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);

    // Process audio in chunks
    for (size_t i = 0; i < audioSamples.size(); i += CHUNK_SIZE) {
        size_t chunkSize = min(CHUNK_SIZE, static_cast<int>(audioSamples.size() - i));
        vector<double> dctTransformed = applyDCT(audioSamples, i, chunkSize, dctMatrix);
        
        // Apply more aggressive quantization
        vector<int16_t> quantized(chunkSize);
        for (size_t j = 0; j < chunkSize; j++) {
            // Progressive quantization based on frequency
            double scale = (j < CHUNK_SIZE/8) ? 1.0 : 
                          ((j < CHUNK_SIZE/4) ? 2.0 : 
                          ((j < CHUNK_SIZE/2) ? 4.0 : 8.0));
                          
            double val = dctTransformed[j] / (QUANTIZATION_FACTOR * scale);
            
            // More aggressive thresholding for higher frequencies
            if (fabs(val) < QUANTIZATION_THRESHOLD * scale) {
                val = 0;
            }
            quantized[j] = static_cast<int16_t>(round(val));
        }
        
        allRleData.push_back(runLengthEncode(quantized));
    }

    // Collect all symbols for Huffman encoding
    vector<int16_t> allSymbols;
    for (const auto& chunk : allRleData) {
        for (const auto& pair : chunk) {
            allSymbols.push_back(pair.first);
        }
    }

    // Apply adaptive Huffman encoding
    AdaptiveHuffman adaptiveHuffman;
    vector<bool> adaptiveHuffmanBits;
    for (int16_t symbol : allSymbols) {
        vector<bool> code = adaptiveHuffman.encode(symbol);
        adaptiveHuffmanBits.insert(adaptiveHuffmanBits.end(), code.begin(), code.end());
    }

    // Save the compressed data
    AudioMetadata metadata;
    metadata.samplerate = sfinfo.samplerate;
    metadata.channels = sfinfo.channels;
    metadata.frames = sfinfo.frames;

    saveCompressedData(allRleData, adaptiveHuffmanBits, metadata, "compressed.bin");
    
    // Calculate compression ratio
    ifstream inFile("compressed.bin", ios::binary | ios::ate);
    size_t compressedSize = inFile.tellg();
    inFile.close();
    
    double ratio = (double)compressedSize / originalSize;
    cout << "Compression complete. Output: compressed.bin (" 
         << compressedSize / 1024.0 << " KB)" << endl;
    cout << "Compression ratio: " << (ratio * 100) << "% of original size" << endl;
}

void decompressAudio(const string& outputFile) {
    ifstream inFile("compressed.bin", ios::binary);
    if (!inFile) throw runtime_error("Error opening compressed file");

    // Read header
    uint32_t headerData[3];
    inFile.read(reinterpret_cast<char*>(headerData), sizeof(headerData));
    
    AudioMetadata metadata;
    metadata.samplerate = headerData[0];
    metadata.channels = headerData[1];
    metadata.frames = headerData[2];

    // Read RLE data
    uint32_t numChunks;
    inFile.read(reinterpret_cast<char*>(&numChunks), sizeof(uint32_t));
    vector<vector<pair<int16_t, uint16_t>>> allRleData(numChunks);

    for (auto& rleData : allRleData) {
        uint32_t numPairs;
        inFile.read(reinterpret_cast<char*>(&numPairs), sizeof(uint32_t));
        rleData.resize(numPairs);
        for (auto& pair : rleData) {
            inFile.read(reinterpret_cast<char*>(&pair.first), sizeof(int16_t));
            inFile.read(reinterpret_cast<char*>(&pair.second), sizeof(uint16_t));
        }
    }

    // Read Huffman encoded data
    uint32_t numBits, numBytes;
    inFile.read(reinterpret_cast<char*>(&numBits), sizeof(uint32_t));
    inFile.read(reinterpret_cast<char*>(&numBytes), sizeof(uint32_t));
    vector<uint8_t> packedBits(numBytes);
    inFile.read(reinterpret_cast<char*>(packedBits.data()), numBytes);
    inFile.close();

    // Unpack and decode
    vector<bool> adaptiveHuffmanBits = unpackBits(packedBits, numBits);

    AdaptiveHuffman adaptiveHuffman;
    vector<int16_t> decodedSymbols;
    size_t pos = 0;
    while (pos < adaptiveHuffmanBits.size()) {
        try {
            decodedSymbols.push_back(adaptiveHuffman.decode(adaptiveHuffmanBits, pos));
        } catch (const runtime_error& e) {
            break; // Stop decoding if we hit an error
        }
    }

    // Map symbols back to RLE data
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

    // Apply inverse DCT
    vector<int16_t> audioSamples;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);
    for (const auto& chunk : allDecodedData) {
        vector<double> dctRestored(chunk.size());
        for (size_t i = 0; i < chunk.size(); i++) {
            // Use the same scaling as in compression
            double scale = (i < CHUNK_SIZE/8) ? 1.0 : 
                          ((i < CHUNK_SIZE/4) ? 2.0 : 
                          ((i < CHUNK_SIZE/2) ? 4.0 : 8.0));
            dctRestored[i] = chunk[i] * QUANTIZATION_FACTOR * scale;
        }
        
        vector<int16_t> signal = inverseDCT(dctRestored, dctMatrix);
        audioSamples.insert(audioSamples.end(), signal.begin(), signal.end());
    }

    // Adjust output size if needed
    size_t expectedSamples = metadata.frames * metadata.channels;
    if (audioSamples.size() > expectedSamples) {
        audioSamples.resize(expectedSamples);
    }

    cout << "Decompressed audio: " << audioSamples.size()/metadata.channels << " frames @ " 
         << metadata.samplerate << " Hz (" 
         << double(audioSamples.size())/metadata.samplerate/metadata.channels << " sec)" << endl;

    // Write output WAV file
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
