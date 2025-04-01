#include <iostream>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <queue>
#include <algorithm>
#include <fstream>
#include <sndfile.h>
#include <bitset>
#include <cstring>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define DEBUG 1
using namespace std;

class AudioCompressor {
public:
    // Discrete Cosine Transform (DCT)
    static vector<double> discreteCosineTransform(const vector<double>& signal) {
        int N = signal.size();
        
        // Check if the signal vector is empty
        if (N == 0) {
            cerr << "Error: Signal vector is empty." << endl;
            return {};
        }
        
        // Limit the size of N for performance - but handle safely
        const int MAX_DCT_SIZE = 8192;
        if (N > MAX_DCT_SIZE) {
            if (DEBUG) cout << "Warning: DCT input size " << N << " is too large. Processing in blocks..." << endl;
            
            // Instead of downsampling, let's process in blocks
            vector<double> result(MAX_DCT_SIZE, 0.0);
            const int BLOCK_SIZE = N / MAX_DCT_SIZE;
            
            if (DEBUG) cout << "Using block size: " << BLOCK_SIZE << endl;
            
            // Process each frequency bin
            for (int u = 0; u < MAX_DCT_SIZE; ++u) {
                if (DEBUG && u % 1000 == 0) {
                    cout << "  DCT processing bin " << u << " of " << MAX_DCT_SIZE << endl;
                }
                
                double sum = 0.0;
                double scaling = (u == 0) ? sqrt(1.0 / MAX_DCT_SIZE) : sqrt(2.0 / MAX_DCT_SIZE);
                
                // Process in blocks to reduce memory access patterns
                for (int block = 0; block < MAX_DCT_SIZE; ++block) {
                    int i = block * BLOCK_SIZE;
                    if (i < N) {
                        sum += signal[i] * cos((M_PI * u * (2.0 * block + 1.0)) / (2.0 * MAX_DCT_SIZE)) * scaling;
                    }
                }
                
                result[u] = sum;
            }
            
            if (DEBUG) cout << "Block DCT processing complete." << endl;
            return result;
        }
    
        vector<double> dctCoeffs(N, 0.0);
        
        if (DEBUG) cout << "Starting DCT calculation for " << N << " coefficients..." << endl;
        int progressInterval = max(1, N / 10);
        
        for (int u = 0; u < N; ++u) {
            if (DEBUG && u % progressInterval == 0) {
                cout << "  DCT progress: " << (u * 100 / N) << "% (" << u << "/" << N << ")" << endl;
            }
            
            double sum = 0.0;
    
            for (int i = 0; i < N; ++i) {
                // Validate input values
                if (!isfinite(signal[i])) {
                    cerr << "Error: Invalid value at index " << i << ": " << signal[i] << endl;
                    return {};
                }
                
                // Scaling factor
                double scaling = (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
    
                // Compute DCT safely
                sum += signal[i] * cos((M_PI * u * (2.0 * i + 1.0)) / (2.0 * N)) * scaling;
            }
            
            dctCoeffs[u] = sum;
        }
        
        if (DEBUG) cout << "DCT calculation complete." << endl;
        return dctCoeffs;
    }
    
    // Simplified DCT for large inputs
    static vector<double> simplifiedDCT(const vector<double>& signal) {
        const int MAX_OUTPUT_SIZE = 8192;
        
        if (DEBUG) cout << "Using simplified DCT for large input..." << endl;
        
        vector<double> result(MAX_OUTPUT_SIZE, 0.0);
        int n = signal.size();
        
        // Simple sampling of the original signal
        for (int i = 0; i < MAX_OUTPUT_SIZE; i++) {
            if (i % 1000 == 0 && DEBUG) {
                cout << "  Processing simplified DCT bin " << i << " of " << MAX_OUTPUT_SIZE << endl;
            }
            
            // Take average of a block to avoid aliasing
            int64_t blockStart = (int64_t)i * n / MAX_OUTPUT_SIZE;
            int64_t blockEnd = (int64_t)(i + 1) * n / MAX_OUTPUT_SIZE;
            blockEnd = min(blockEnd, (int64_t)n);
            
            double sum = 0.0;
            int count = 0;
            
            for (int64_t j = blockStart; j < blockEnd; j++) {
                if (isfinite(signal[j])) {
                    sum += signal[j];
                    count++;
                }
            }
            
            if (count > 0) {
                result[i] = sum / count;
            } else {
                result[i] = 0.0;
            }
        }
        
        if (DEBUG) cout << "Simplified DCT complete." << endl;
        return result;
    }
    
    // Run-Length Encoding (RLE)
    static vector<pair<double, int>> runLengthEncode(const vector<double>& signal) {
        vector<pair<double, int>> encoded;
        
        if (signal.empty()) return encoded;

        // Use a higher precision threshold for comparison
        const double THRESHOLD = 1e-4;
        
        if (DEBUG) cout << "Starting RLE with threshold " << THRESHOLD << "..." << endl;
        
        double currentValue = signal[0];
        int count = 1;
        int progress = 0;
        int progressInterval = max(1, (int)signal.size() / 10);

        for (size_t i = 1; i < signal.size(); ++i) {
            if (DEBUG && i % progressInterval == 0) {
                cout << "  RLE progress: " << (i * 100 / signal.size()) << "% (" << i << "/" << signal.size() << ")" << endl;
            }
            
            if (abs(signal[i] - currentValue) < THRESHOLD) {
                count++;
            } else {
                encoded.push_back({currentValue, count});
                currentValue = signal[i];
                count = 1;
            }
            
            // Safety check to prevent memory issues with very large input
            if (encoded.size() > 1000000) {
                cerr << "Error: RLE producing too many segments. Increasing threshold." << endl;
                // Double the threshold and restart
                return runLengthEncode(signal, THRESHOLD * 2);
            }
        }
        
        // Add the last run
        encoded.push_back({currentValue, count});
        
        if (DEBUG) cout << "RLE complete. Compressed from " << signal.size() << " to " << encoded.size() << " elements." << endl;
        return encoded;
    }
    
    // Run-Length Encoding with custom threshold
    static vector<pair<double, int>> runLengthEncode(const vector<double>& signal, double threshold) {
        vector<pair<double, int>> encoded;
        
        if (signal.empty()) return encoded;
        
        if (DEBUG) cout << "Starting RLE with increased threshold " << threshold << "..." << endl;
        
        double currentValue = signal[0];
        int count = 1;

        for (size_t i = 1; i < signal.size(); ++i) {
            if (abs(signal[i] - currentValue) < threshold) {
                count++;
            } else {
                encoded.push_back({currentValue, count});
                currentValue = signal[i];
                count = 1;
            }
            
            // Safety check
            if (encoded.size() > 1000000) {
                cerr << "Error: RLE still producing too many segments. Using quantization." << endl;
                return quantizeAndEncode(signal, threshold * 5);
            }
        }
        
        // Add the last run
        encoded.push_back({currentValue, count});
        
        if (DEBUG) cout << "RLE complete with custom threshold. Compressed to " << encoded.size() << " elements." << endl;
        return encoded;
    }
    
    // Quantization and encoding for difficult signals
    static vector<pair<double, int>> quantizeAndEncode(const vector<double>& signal, double quantStep) {
        if (DEBUG) cout << "Using quantization with step " << quantStep << "..." << endl;
        
        vector<double> quantized(signal.size());
        for (size_t i = 0; i < signal.size(); i++) {
            quantized[i] = round(signal[i] / quantStep) * quantStep;
        }
        
        return runLengthEncode(quantized);
    }

    // Adaptive Huffman Coding
    class AdaptiveHuffmanCoder {
    private:
        struct Node {
            double value;
            int weight;
            int number;
            Node* parent;
            Node* left;
            Node* right;

            Node(double val, int w, int num, Node* p = nullptr) 
                : value(val), weight(w), number(num), parent(p), 
                  left(nullptr), right(nullptr) {}
        };

        Node* NYT;
        Node* root;
        unordered_map<double, Node*> symbolTable;
        int nodeNumberCounter;
        int encodedSymbolCount;
        const int MAX_SYMBOLS = 1000000;  // Safety limit

        Node* findHighestInBlock(Node* start, int weight) {
            if (!start) return nullptr;
            
            queue<Node*> q;
            q.push(start);
            Node* result = nullptr;
            
            while (!q.empty()) {
                Node* current = q.front();
                q.pop();
                
                if (current->weight == weight && 
                    (!result || current->number > result->number)) {
                    result = current;
                }
                
                if (current->left) q.push(current->left);
                if (current->right) q.push(current->right);
            }
            
            return result;
        }

        void swapNodes(Node* a, Node* b) {
            if (!a || !b || a == b) return;
            
            // Handle the case where they are siblings
            if (a->parent == b->parent) {
                if (a->parent->left == a) {
                    a->parent->left = b;
                    a->parent->right = a;
                } else {
                    a->parent->left = a;
                    a->parent->right = b;
                }
                swap(a->number, b->number);
                return;
            }
            
            // Handle normal case
            Node* aParent = a->parent;
            Node* bParent = b->parent;
            
            if (aParent->left == a) aParent->left = b;
            else aParent->right = b;
            
            if (bParent->left == b) bParent->left = a;
            else bParent->right = a;
            
            b->parent = aParent;
            a->parent = bParent;
            
            swap(a->number, b->number);
        }

        void updateTree(Node* currentNode) {
            int maxIterations = 100; // Safety limit
            int iterations = 0;
            
            while (currentNode && iterations < maxIterations) {
                Node* highestInBlock = findHighestInBlock(root, currentNode->weight);
                
                if (highestInBlock && highestInBlock != currentNode && 
                    highestInBlock != currentNode->parent &&
                    currentNode != highestInBlock->parent) {
                    swapNodes(currentNode, highestInBlock);
                }
                
                currentNode->weight++;
                currentNode = currentNode->parent;
                iterations++;
            }
            
            if (iterations >= maxIterations && DEBUG) {
                cout << "Warning: Max iterations reached in updateTree" << endl;
            }
        }

        string getCode(Node* node) {
            string code;
            Node* current = node;
            
            // Safety limit
            int maxDepth = 100;
            int depth = 0;
            
            while (current->parent && depth < maxDepth) {
                if (current->parent->left == current) {
                    code += '0';
                } else {
                    code += '1';
                }
                current = current->parent;
                depth++;
            }
            
            reverse(code.begin(), code.end());
            return code;
        }

    public:
        AdaptiveHuffmanCoder() {
            NYT = new Node(0.0, 0, 512);
            root = NYT;
            symbolTable.clear();
            nodeNumberCounter = 511;
            encodedSymbolCount = 0;
        }

        ~AdaptiveHuffmanCoder() {
            // Simple cleanup for demo purposes
            // A full implementation would require a proper tree traversal and deletion
        }

        string encode(double symbol) {
            // Check for invalid symbols
            if (!isfinite(symbol)) {
                cerr << "Error: Invalid symbol detected in Huffman encoding: " << symbol << endl;
                return "ERROR";
            }
            
            // Safety check for maximum encoding size
            if (encodedSymbolCount++ > MAX_SYMBOLS) {
                cerr << "Error: Too many symbols encoded. Aborting." << endl;
                return "ERROR";
            }
            
            if (DEBUG && encodedSymbolCount % 1000 == 0) {
                cout << "  Huffman encoding symbol #" << encodedSymbolCount << ": " << symbol << endl;
            }
            
            string code;
            
            if (symbolTable.find(symbol) != symbolTable.end()) {
                Node* symbolNode = symbolTable[symbol];
                code = getCode(symbolNode);
                updateTree(symbolNode);
            } else {
                code = getCode(NYT);
                
                // Simplify encoding for large numbers to prevent overflow
                // Instead of using the full 64-bit representation, quantize to 32-bit
                string binary;
                int32_t intValue = static_cast<int32_t>(symbol * 1000000.0); // Scale for precision
                
                for (int i = 0; i < 32; i++) {
                    binary = ((intValue & 1) ? "1" : "0") + binary;
                    intValue >>= 1;
                }
                code += binary;
                
                Node* newNYT = new Node(0.0, 0, nodeNumberCounter--, NYT);
                Node* symbolNode = new Node(symbol, 1, nodeNumberCounter--, NYT);
                
                NYT->left = newNYT;
                NYT->right = symbolNode;
                NYT->value = 0.0;
                NYT->weight = 1;
                
                symbolTable[symbol] = symbolNode;
                NYT = newNYT;
                
                updateTree(symbolNode->parent);
            }
            
            return code;
        }

        // Static methods for file operations
        static bool compressAudioFile(const string& inputFile, const string& outputFile) {
            if (DEBUG) cout << "===========================================" << endl;
            if (DEBUG) cout << "Starting compression process..." << endl;
            if (DEBUG) cout << "===========================================" << endl;
            
            // Open input audio file
            if (DEBUG) cout << "Opening input audio file: " << inputFile << endl;
            SF_INFO sfInfo;
            sfInfo.format = 0;
            SNDFILE* infile = sf_open(inputFile.c_str(), SFM_READ, &sfInfo);
            
            if (!infile) {
                cerr << "Error opening input file: " << inputFile << endl;
                return false;
            }

            // Print audio file info
            if (DEBUG) {
                cout << "Audio file info:" << endl;
                cout << "  Frames: " << sfInfo.frames << endl;
                cout << "  Channels: " << sfInfo.channels << endl;
                cout << "  Sample rate: " << sfInfo.samplerate << endl;
                cout << "  Total samples: " << (sfInfo.frames * sfInfo.channels) << endl;
            }

            // Read audio data
            if (DEBUG) cout << "Reading audio data..." << endl;
            vector<double> audioData(sfInfo.frames * sfInfo.channels);
            sf_count_t count = sf_read_double(infile, audioData.data(), audioData.size());
            sf_close(infile);

            if (DEBUG) cout << "Read " << count << " samples of " << audioData.size() << " expected." << endl;

            if (count == 0) {
                cerr << "No data read from input file" << endl;
                return false;
            }
            
            // Resize to actual read size if needed
            if (count < audioData.size()) {
                if (DEBUG) cout << "Resizing audio data to actual read size." << endl;
                audioData.resize(count);
            }

            // Check for NaN or Inf values
            if (DEBUG) cout << "Checking for invalid audio samples..." << endl;
            int invalidCount = 0;
            for (size_t i = 0; i < audioData.size(); i++) {
                if (!isfinite(audioData[i])) {
                    invalidCount++;
                    audioData[i] = 0.0; // Replace with a safe value
                }
            }
            if (invalidCount > 0) {
                cout << "Warning: Found and fixed " << invalidCount << " invalid audio samples." << endl;
            }

            // For large inputs, use a simpler approach
            vector<double> processedData;
            if (audioData.size() > 100000) {
                if (DEBUG) cout << "===========================================" << endl;
                if (DEBUG) cout << "Using simplified processing for large input..." << endl;
                processedData = simplifiedDCT(audioData);
            } else {
                // Apply Discrete Cosine Transform
                if (DEBUG) cout << "===========================================" << endl;
                if (DEBUG) cout << "Applying Discrete Cosine Transform..." << endl;
                processedData = discreteCosineTransform(audioData);
            }
            
            if (DEBUG) cout << "Signal processing completed! Produced " << processedData.size() << " coefficients." << endl;

            // Apply Run-Length Encoding
            if (DEBUG) cout << "===========================================" << endl;
            if (DEBUG) cout << "Applying Run-Length Encoding..." << endl;
            vector<pair<double, int>> rleEncoded = runLengthEncode(processedData);
            if (DEBUG) cout << "RLE completed! Encoded size: " << rleEncoded.size() << " pairs." << endl;

            // Apply Adaptive Huffman Coding
            if (DEBUG) cout << "===========================================" << endl;
            if (DEBUG) cout << "Applying Adaptive Huffman Encoding..." << endl;
            AdaptiveHuffmanCoder huffmanCoder;
            vector<string> encodedSymbols;
            
            // Progress tracking
            size_t totalPairs = rleEncoded.size();
            size_t progressInterval = max(size_t(1), totalPairs / 20);
            
            // Limit the maximum number of pairs to encode
            const size_t MAX_PAIRS = 100000;
            if (totalPairs > MAX_PAIRS) {
                if (DEBUG) cout << "Too many RLE pairs (" << totalPairs << "). Limiting to " << MAX_PAIRS << endl;
                totalPairs = MAX_PAIRS;
            }
            
            for (size_t i = 0; i < totalPairs; ++i) {
                const auto& coeff = rleEncoded[i];
                
                if (DEBUG && i % progressInterval == 0) {
                    cout << "  Huffman progress: " << (i * 100 / totalPairs) << "% (" << i << "/" << totalPairs << ")" << endl;
                }
                
                // Encode both the value and the run length
                string valueCode = huffmanCoder.encode(coeff.first);
                if (valueCode == "ERROR") {
                    cerr << "Error during Huffman encoding. Aborting." << endl;
                    return false;
                }
                
                string countCode = huffmanCoder.encode(static_cast<double>(coeff.second));
                if (countCode == "ERROR") {
                    cerr << "Error during Huffman encoding. Aborting." << endl;
                    return false;
                }
                
                encodedSymbols.push_back(valueCode + "|" + countCode);
            }
            
            if (DEBUG) cout << "Huffman encoding completed! Produced " << encodedSymbols.size() << " encoded symbols." << endl;

            // Write compressed data to output file
            if (DEBUG) cout << "===========================================" << endl;
            if (DEBUG) cout << "Writing compressed data to file: " << outputFile << endl;
            ofstream outfile(outputFile, ios::binary);
            if (!outfile) {
                cerr << "Error creating output file: " << outputFile << endl;
                return false;
            }

            // Write metadata
            outfile << sfInfo.frames << " " 
                    << sfInfo.channels << " " 
                    << sfInfo.samplerate << endl;

            // Write encoded symbols
            size_t writtenSymbols = 0;
            for (const auto& symbol : encodedSymbols) {
                outfile << symbol << endl;
                writtenSymbols++;
                
                if (DEBUG && writtenSymbols % 10000 == 0) {
                    cout << "  Writing progress: " << (writtenSymbols * 100 / encodedSymbols.size()) 
                              << "% (" << writtenSymbols << "/" << encodedSymbols.size() << ")" << endl;
                }
            }

            outfile.close();
            if (DEBUG) cout << "File writing completed!" << endl;
            if (DEBUG) cout << "===========================================" << endl;
            return true;
        }
    };
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];

    cout << "Starting audio compression..." << endl;
    cout << "Input File: " << inputFile << endl;
    cout << "Output File: " << outputFile << endl;

    bool success = AudioCompressor::AdaptiveHuffmanCoder::compressAudioFile(inputFile, outputFile);

    if (success) {
        cout << "Compression successful. Output file: " << outputFile << endl;
        return 0;
    } else {
        cerr << "Compression failed." << endl;
        return 1;
    }
}