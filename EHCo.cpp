#include <iostream>
#include <vector>
#include <map>
#include <queue>
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <sndfile.h>
#include <functional>
#include <thread>
#include <future>

using namespace std;

const int CHUNK_SIZE = 4096;
const int QUANTIZATION_FACTOR = 256;
const double QUANTIZATION_THRESHOLD = 0.7;

struct AudioMetadata {
    int samplerate, channels;
    sf_count_t frames;
};

vector<vector<double>> precomputeDCTMatrix(int N) {
    vector<vector<double>> dctMatrix(N, vector<double>(N));
    double sqrtN = sqrt(1.0/N);
    double sqrt2N = sqrt(2.0/N);
    
    for (int k = 0; k < N; ++k) {
        double scale = (k == 0) ? sqrtN : sqrt2N;
        double angle = M_PI * k / (2.0 * N);
        for (int n = 0; n < N; ++n) {
            dctMatrix[k][n] = scale * cos(angle * (2 * n + 1));
        }
    }
    return dctMatrix;
}

vector<double> applyDCT(const vector<int16_t>& signal, int start, int length, const vector<vector<double>>& dctMatrix) {
    vector<double> transformed(length);
    for (int k = 0; k < length; ++k) {
        double sum = 0.0;
        for (int n = 0; n < length; ++n) {
            sum += signal[start + n] * dctMatrix[k][n];
        }
        transformed[k] = sum;
    }
    return transformed;
}

vector<int16_t> inverseDCT(const vector<double>& transformed, const vector<vector<double>>& dctMatrix) {
    int length = transformed.size();
    vector<int16_t> signal(length);
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
    
    int16_t current = data[0];
    uint16_t count = 1;
    
    for (size_t i = 1; i < data.size(); ++i) {
        if (data[i] == current && count < numeric_limits<uint16_t>::max()) {
            ++count;
        } else {
            encoded.emplace_back(current, count);
            current = data[i];
            count = 1;
        }
    }
    encoded.emplace_back(current, count);
    return encoded;
}

vector<int16_t> runLengthDecode(const vector<pair<int16_t, uint16_t>>& encoded) {
    vector<int16_t> decoded;
    for (const auto& p : encoded) {
        decoded.insert(decoded.end(), p.second, p.first);
    }
    return decoded;
}

struct HuffmanNode {
    int16_t symbol;
    int weight, order;
    HuffmanNode *parent, *left, *right;

    HuffmanNode(int16_t sym, int wt, int ord) 
        : symbol(sym), weight(wt), order(ord), parent(nullptr), left(nullptr), right(nullptr) {}
};

class AdaptiveHuffman {
    map<int16_t, HuffmanNode*> symbolMap;
    HuffmanNode* root;
    int nextOrder;
    const int16_t NYT_SYMBOL = numeric_limits<int16_t>::max();

    vector<bool> getCode(HuffmanNode* node) {
        vector<bool> code;
        while (node->parent) {
            code.push_back(node == node->parent->right);
            node = node->parent;
        }
        reverse(code.begin(), code.end());
        return code;
    }

    void swapNodes(HuffmanNode* a, HuffmanNode* b) {
        if (!a->parent || !b->parent) return;
        
        if (a->parent->left == a) a->parent->left = b;
        else a->parent->right = b;
        
        if (b->parent->left == b) b->parent->left = a;
        else b->parent->right = a;
        
        swap(a->parent, b->parent);
        swap(a->order, b->order);
    }

    HuffmanNode* findNodeToSwap(HuffmanNode* node) {
        HuffmanNode* candidate = nullptr;
        queue<HuffmanNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            HuffmanNode* current = q.front();
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

    void updateTree(int16_t symbol) {
        HuffmanNode* node;
        
        if (symbolMap.find(symbol) != symbolMap.end()) {
            node = symbolMap[symbol];
        } else {
            HuffmanNode* nytNode = symbolMap[NYT_SYMBOL];
            HuffmanNode* internal = new HuffmanNode(0, 0, nextOrder--);
            internal->parent = nytNode->parent;
            
            if (nytNode->parent) {
                (nytNode->parent->left == nytNode ? nytNode->parent->left : nytNode->parent->right) = internal;
            } else {
                root = internal;
            }
            
            HuffmanNode* newNYT = new HuffmanNode(NYT_SYMBOL, 0, nextOrder--);
            newNYT->parent = internal;
            internal->left = newNYT;
            symbolMap[NYT_SYMBOL] = newNYT;
            
            node = new HuffmanNode(symbol, 0, nextOrder--);
            node->parent = internal;
            internal->right = node;
            symbolMap[symbol] = node;
        }
        
        while (node) {
            HuffmanNode* toSwap = findNodeToSwap(node);
            if (toSwap && node != toSwap && node->parent != toSwap) {
                swapNodes(node, toSwap);
            }
            node->weight++;
            node = node->parent;
        }
    }

public:
    AdaptiveHuffman() : nextOrder(511) {
        root = new HuffmanNode(NYT_SYMBOL, 0, 512);
        symbolMap[NYT_SYMBOL] = root;
    }

    ~AdaptiveHuffman() {
        function<void(HuffmanNode*)> freeTree = [&](HuffmanNode* node) {
            if (!node) return;
            freeTree(node->left);
            freeTree(node->right);
            delete node;
        };
        freeTree(root);
    }

    vector<bool> encode(int16_t symbol) {
        vector<bool> code;
        if (symbolMap.find(symbol) != symbolMap.end()) {
            code = getCode(symbolMap[symbol]);
        } else {
            vector<bool> nytCode = getCode(symbolMap[NYT_SYMBOL]);
            code.insert(code.end(), nytCode.begin(), nytCode.end());
            
            uint16_t absSym = abs(symbol);
            int bitWidth = 0;
            while ((1 << bitWidth) <= absSym) bitWidth++;
            
            for (int i = 3; i >= 0; i--) code.push_back((bitWidth >> i) & 1);
            code.push_back(symbol < 0);
            for (int i = bitWidth - 1; i >= 0; i--) code.push_back((absSym >> i) & 1);
        }
        updateTree(symbol);
        return code;
    }

    int16_t decode(const vector<bool>& code, size_t& pos) {
        HuffmanNode* node = root;
        while (node->left || node->right) {
            if (pos >= code.size()) throw runtime_error("Invalid code");
            node = code[pos++] ? node->right : node->left;
        }
        
        if (node->symbol == NYT_SYMBOL) {
            int bitWidth = 0;
            for (int i = 0; i < 4; i++) bitWidth = (bitWidth << 1) | (code[pos++] ? 1 : 0);
            bool isNeg = code[pos++];
            uint16_t mag = 0;
            for (int i = 0; i < bitWidth; i++) mag = (mag << 1) | (code[pos++] ? 1 : 0);
            updateTree(isNeg ? -mag : mag);
            return isNeg ? -mag : mag;
        }
        
        updateTree(node->symbol);
        return node->symbol;
    }
};

vector<uint8_t> packBits(const vector<bool>& bits) {
    vector<uint8_t> packed;
    uint8_t byte = 0;
    int bitPos = 7;
    
    for (bool bit : bits) {
        if (bit) byte |= (1 << bitPos);
        if (--bitPos < 0) {
            packed.push_back(byte);
            byte = 0;
            bitPos = 7;
        }
    }
    if (bitPos != 7) packed.push_back(byte);
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

void saveCompressedData(const vector<vector<pair<int16_t, uint16_t>>>& rleData,
                      const vector<bool>& huffmanBits,
                      const AudioMetadata& metadata,
                      const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out) throw runtime_error("Failed to create file");
    
    out.write(reinterpret_cast<const char*>(&metadata), sizeof(metadata));
    uint32_t numChunks = rleData.size();
    out.write(reinterpret_cast<const char*>(&numChunks), sizeof(numChunks));
    
    for (const auto& chunk : rleData) {
        uint32_t numPairs = chunk.size();
        out.write(reinterpret_cast<const char*>(&numPairs), sizeof(numPairs));
        out.write(reinterpret_cast<const char*>(chunk.data()), numPairs * sizeof(pair<int16_t, uint16_t>));
    }
    
    auto packed = packBits(huffmanBits);
    uint32_t numBits = huffmanBits.size(), numBytes = packed.size();
    out.write(reinterpret_cast<const char*>(&numBits), sizeof(numBits));
    out.write(reinterpret_cast<const char*>(&numBytes), sizeof(numBytes));
    out.write(reinterpret_cast<const char*>(packed.data()), numBytes);
}

void compressAudio(const string& filename) {
    SF_INFO sfinfo = {};
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) throw runtime_error("Error opening file");

    vector<int16_t> samples(sfinfo.frames * sfinfo.channels);
    sf_read_short(file, samples.data(), samples.size());
    sf_close(file);

    size_t originalSize = samples.size() * sizeof(int16_t);
    cout << "Original: " << originalSize/1024.0 << " KB\n";

    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);
    size_t numChunks = (samples.size() + CHUNK_SIZE - 1) / CHUNK_SIZE;
    vector<vector<pair<int16_t, uint16_t>>> allRleData(numChunks);
    vector<vector<int16_t>> allSymbolsChunks(numChunks);

    unsigned int maxThreads = thread::hardware_concurrency();
    if (maxThreads == 0) maxThreads = 4;
    vector<future<void>> futures;

    for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        if (futures.size() >= maxThreads) {
            futures.front().get();
            futures.erase(futures.begin());
        }
        futures.emplace_back(async(launch::async, [&, chunkIdx]() {
            size_t i = chunkIdx * CHUNK_SIZE;
            size_t chunkSize = min(static_cast<size_t>(CHUNK_SIZE), samples.size() - i);
            auto t = applyDCT(samples, i, chunkSize, dctMatrix);
            vector<int16_t> quantized(chunkSize);
            for (size_t j = 0; j < chunkSize; j++) {
                double scale = (j < CHUNK_SIZE/8) ? 1.0 : 
                             ((j < CHUNK_SIZE/4) ? 2.0 : 
                             ((j < CHUNK_SIZE/2) ? 4.0 : 8.0));
                double val = t[j] / (QUANTIZATION_FACTOR * scale);
                if (fabs(val) < QUANTIZATION_THRESHOLD * scale) val = 0;
                quantized[j] = static_cast<int16_t>(round(val));
            }
            auto rle = runLengthEncode(quantized);
            allRleData[chunkIdx] = rle;
            allSymbolsChunks[chunkIdx].reserve(rle.size());
            for (const auto& p : rle) allSymbolsChunks[chunkIdx].push_back(p.first);
        }));
    }
    for (auto& fut : futures) fut.get();

    // Flatten allSymbolsChunks into allSymbols
    vector<int16_t> allSymbols;
    size_t totalSymbols = 0;
    for (const auto& v : allSymbolsChunks) totalSymbols += v.size();
    allSymbols.reserve(totalSymbols);
    for (const auto& v : allSymbolsChunks) allSymbols.insert(allSymbols.end(), v.begin(), v.end());

    AdaptiveHuffman huffman;
    vector<bool> bits;
    bits.reserve(allSymbols.size() * 16); // Reserve a rough upper bound
    for (int16_t sym : allSymbols) {
        auto code = huffman.encode(sym);
        bits.insert(bits.end(), code.begin(), code.end());
    }

    AudioMetadata metadata = {sfinfo.samplerate, sfinfo.channels, sfinfo.frames};
    saveCompressedData(allRleData, bits, metadata, "compressed.bin");

    ifstream in("compressed.bin", ios::binary | ios::ate);
    size_t compressedSize = in.tellg();
    cout << "Compressed: " << compressedSize/1024.0 << " KB (" 
         << (100.0 * compressedSize / originalSize) << "%)\n";
}

void decompressAudio(const string& outputFile) {
    ifstream in("compressed.bin", ios::binary);
    if (!in) throw runtime_error("Error opening file");

    AudioMetadata metadata;
    in.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));

    uint32_t numChunks;
    in.read(reinterpret_cast<char*>(&numChunks), sizeof(numChunks));
    vector<vector<pair<int16_t, uint16_t>>> rleData(numChunks);

    for (auto& chunk : rleData) {
        uint32_t numPairs;
        in.read(reinterpret_cast<char*>(&numPairs), sizeof(numPairs));
        chunk.resize(numPairs);
        in.read(reinterpret_cast<char*>(chunk.data()), numPairs * sizeof(pair<int16_t, uint16_t>));
    }

    uint32_t numBits, numBytes;
    in.read(reinterpret_cast<char*>(&numBits), sizeof(numBits));
    in.read(reinterpret_cast<char*>(&numBytes), sizeof(numBytes));
    vector<uint8_t> packed(numBytes);
    in.read(reinterpret_cast<char*>(packed.data()), numBytes);

    auto bits = unpackBits(packed, numBits);
    AdaptiveHuffman huffman;
    vector<int16_t> symbols;
    symbols.reserve(numBits / 8); // Reserve a rough estimate
    size_t pos = 0;

    while (pos < bits.size()) {
        try {
            symbols.push_back(huffman.decode(bits, pos));
        } catch (...) { break; }
    }

    size_t symIdx = 0;
    auto dctMatrix = precomputeDCTMatrix(CHUNK_SIZE);
    vector<vector<int16_t>> outputChunks(numChunks);
    vector<future<void>> futures;
    unsigned int maxThreads = thread::hardware_concurrency();
    if (maxThreads == 0) maxThreads = 4;

    // Precompute chunk start indices for symbols vector (by RLE pairs, not by total samples)
    vector<size_t> chunkSymbolStart(numChunks+1, 0);
    size_t tempSymIdx = 0;
    for (size_t ci = 0; ci < numChunks; ++ci) {
        chunkSymbolStart[ci] = tempSymIdx;
        tempSymIdx += rleData[ci].size();
    }
    chunkSymbolStart[numChunks] = tempSymIdx;

    for (size_t chunkIdx = 0; chunkIdx < numChunks; ++chunkIdx) {
        if (futures.size() >= maxThreads) {
            futures.front().get();
            futures.erase(futures.begin());
        }
        futures.emplace_back(async(launch::async, [&, chunkIdx]() {
            vector<int16_t> quantized;
            quantized.reserve(CHUNK_SIZE);
            size_t localSymIdx = chunkSymbolStart[chunkIdx];
            for (const auto& p : rleData[chunkIdx]) {
                if (localSymIdx >= symbols.size()) throw runtime_error("Symbol mismatch");
                quantized.insert(quantized.end(), p.second, symbols[localSymIdx]);
                localSymIdx++;
            }
            vector<double> restored(quantized.size());
            for (size_t i = 0; i < quantized.size(); i++) {
                double scale = (i < CHUNK_SIZE/8) ? 1.0 : 
                             ((i < CHUNK_SIZE/4) ? 2.0 : 
                             ((i < CHUNK_SIZE/2) ? 4.0 : 8.0));
                restored[i] = quantized[i] * QUANTIZATION_FACTOR * scale;
            }
            outputChunks[chunkIdx] = inverseDCT(restored, dctMatrix);
        }));
    }
    for (auto& fut : futures) fut.get();

    // Flatten outputChunks into output
    vector<int16_t> output;
    size_t totalOutput = 0;
    for (const auto& v : outputChunks) totalOutput += v.size();
    output.reserve(totalOutput);
    for (const auto& v : outputChunks) output.insert(output.end(), v.begin(), v.end());

    output.resize(metadata.frames * metadata.channels);

    SF_INFO sfinfo = {};
    sfinfo.samplerate = metadata.samplerate;
    sfinfo.channels = metadata.channels;
    sfinfo.frames = output.size() / metadata.channels;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outFile = sf_open(outputFile.c_str(), SFM_WRITE, &sfinfo);
    if (!outFile) throw runtime_error("Error creating output");
    sf_write_short(outFile, output.data(), output.size());
    sf_close(outFile);

    cout << "Decompressed to " << outputFile << endl;
}

int main(int argc, char* argv[]) {
    try {
        if (argc > 1 && strcmp(argv[1], "-d") == 0) {
            decompressAudio("output.wav");
        } else if (argc > 1) {
            compressAudio(argv[1]);
        } else {
            cerr << "Usage: " << argv[0] << " <input.wav> (compress)\n"
                 << "       " << argv[0] << " -d (decompress)\n";
            return 1;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    return 0;
}
