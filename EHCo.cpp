#include <iostream>
#include <vector>
#include <map>
#include <queue>
#define _USE_MATH_DEFINES
#include <cmath>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <sndfile.h>
#include <functional>
#include <thread>
#include <future>
#include "compression_algorithms.cpp"

using namespace std;

const int CHUNK_SIZE = 4096;
const int QUANTIZATION_FACTOR = 200;
const double QUANTIZATION_THRESHOLD = 0.1;

struct AudioMetadata {
    int samplerate, channels;
    sf_count_t frames;
};

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
    
    auto packed = Huffman::packBits(huffmanBits);
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

    auto dctMatrix = DCT::precomputeDCTMatrix(CHUNK_SIZE);
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
            auto t = DCT::apply(samples, i, chunkSize, dctMatrix);
            vector<int16_t> quantized(chunkSize);
            for (size_t j = 0; j < chunkSize; j++) {
                double scale = (j < CHUNK_SIZE/8) ? 1.0 : 
                             ((j < CHUNK_SIZE/4) ? 2.0 : 
                             ((j < CHUNK_SIZE/2) ? 4.0 : 8.0));
                double val = t[j] / (QUANTIZATION_FACTOR * scale);
                if (fabs(val) < QUANTIZATION_THRESHOLD * scale) val = 0;
                quantized[j] = static_cast<int16_t>(round(val));
            }
            auto rle = RLE::encode(quantized);
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

    Huffman::AdaptiveHuffman huffman;
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

    auto bits = Huffman::unpackBits(packed, numBits);
    Huffman::AdaptiveHuffman huffman;
    vector<int16_t> symbols;
    symbols.reserve(numBits / 8); // Reserve a rough estimate
    size_t pos = 0;

    while (pos < bits.size()) {
        try {
            symbols.push_back(huffman.decode(bits, pos));
        } catch (...) { break; }
    }

    size_t symIdx = 0;
    auto dctMatrix = DCT::precomputeDCTMatrix(CHUNK_SIZE);
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
            outputChunks[chunkIdx] = DCT::inverse(restored, dctMatrix);
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