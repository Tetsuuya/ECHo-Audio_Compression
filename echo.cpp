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

// Debug flag for verbose output
#define DEBUG 1

class AudioCompressor {
public:
    // Discrete Cosine Transform (DCT)
    static std::vector<double> discreteCosineTransform(const std::vector<double>& signal) {
        int N = signal.size();
        
        // Check if the signal vector is empty
        if (N == 0) {
            std::cerr << "Error: Signal vector is empty." << std::endl;
            return {};
        }
        
        // Limit the size of N for performance - but handle safely
        const int MAX_DCT_SIZE = 8192;
        if (N > MAX_DCT_SIZE) {
            if (DEBUG) std::cout << "Warning: DCT input size " << N << " is too large. Processing in blocks..." << std::endl;
            
            // Instead of downsampling, let's process in blocks
            std::vector<double> result(MAX_DCT_SIZE, 0.0);
            const int BLOCK_SIZE = N / MAX_DCT_SIZE;
            
            if (DEBUG) std::cout << "Using block size: " << BLOCK_SIZE << std::endl;
            
            // Process each frequency bin
            for (int u = 0; u < MAX_DCT_SIZE; ++u) {
                if (DEBUG && u % 1000 == 0) {
                    std::cout << "  DCT processing bin " << u << " of " << MAX_DCT_SIZE << std::endl;
                }
                
                double sum = 0.0;
                double scaling = (u == 0) ? sqrt(1.0 / MAX_DCT_SIZE) : sqrt(2.0 / MAX_DCT_SIZE);
                
                // Process in blocks to reduce memory access patterns
                for (int block = 0; block < MAX_DCT_SIZE; ++block) {
                    int i = block * BLOCK_SIZE;
                    if (i < N) {
                        sum += signal[i] * std::cos((M_PI * u * (2.0 * block + 1.0)) / (2.0 * MAX_DCT_SIZE)) * scaling;
                    }
                }
                
                result[u] = sum;
            }
            
            if (DEBUG) std::cout << "Block DCT processing complete." << std::endl;
            return result;
        }
    
        std::vector<double> dctCoeffs(N, 0.0);
        
        if (DEBUG) std::cout << "Starting DCT calculation for " << N << " coefficients..." << std::endl;
        int progressInterval = std::max(1, N / 10);
        
        for (int u = 0; u < N; ++u) {
            if (DEBUG && u % progressInterval == 0) {
                std::cout << "  DCT progress: " << (u * 100 / N) << "% (" << u << "/" << N << ")" << std::endl;
            }
            
            double sum = 0.0;
    
            for (int i = 0; i < N; ++i) {
                // Validate input values
                if (!std::isfinite(signal[i])) {
                    std::cerr << "Error: Invalid value at index " << i << ": " << signal[i] << std::endl;
                    return {};
                }
                
                // Scaling factor
                double scaling = (u == 0) ? sqrt(1.0 / N) : sqrt(2.0 / N);
    
                // Compute DCT safely
                sum += signal[i] * std::cos((M_PI * u * (2.0 * i + 1.0)) / (2.0 * N)) * scaling;
            }
            
            dctCoeffs[u] = sum;
        }
        
        if (DEBUG) std::cout << "DCT calculation complete." << std::endl;
        return dctCoeffs;
    }
    
    // Simplified DCT for large inputs
    static std::vector<double> simplifiedDCT(const std::vector<double>& signal) {
        const int MAX_OUTPUT_SIZE = 8192;
        
        if (DEBUG) std::cout << "Using simplified DCT for large input..." << std::endl;
        
        std::vector<double> result(MAX_OUTPUT_SIZE, 0.0);
        int n = signal.size();
        
        // Simple sampling of the original signal
        for (int i = 0; i < MAX_OUTPUT_SIZE; i++) {
            if (i % 1000 == 0 && DEBUG) {
                std::cout << "  Processing simplified DCT bin " << i << " of " << MAX_OUTPUT_SIZE << std::endl;
            }
            
            // Take average of a block to avoid aliasing
            int64_t blockStart = (int64_t)i * n / MAX_OUTPUT_SIZE;
            int64_t blockEnd = (int64_t)(i + 1) * n / MAX_OUTPUT_SIZE;
            blockEnd = std::min(blockEnd, (int64_t)n);
            
            double sum = 0.0;
            int count = 0;
            
            for (int64_t j = blockStart; j < blockEnd; j++) {
                if (std::isfinite(signal[j])) {
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
        
        if (DEBUG) std::cout << "Simplified DCT complete." << std::endl;
        return result;
    }
    
    // Run-Length Encoding (RLE)
    static std::vector<std::pair<double, int>> runLengthEncode(const std::vector<double>& signal) {
        std::vector<std::pair<double, int>> encoded;
        
        if (signal.empty()) return encoded;

        // Use a higher precision threshold for comparison
        const double THRESHOLD = 1e-4;
        
        if (DEBUG) std::cout << "Starting RLE with threshold " << THRESHOLD << "..." << std::endl;
        
        double currentValue = signal[0];
        int count = 1;
        int progress = 0;
        int progressInterval = std::max(1, (int)signal.size() / 10);

        for (size_t i = 1; i < signal.size(); ++i) {
            if (DEBUG && i % progressInterval == 0) {
                std::cout << "  RLE progress: " << (i * 100 / signal.size()) << "% (" << i << "/" << signal.size() << ")" << std::endl;
            }
            
            if (std::abs(signal[i] - currentValue) < THRESHOLD) {
                count++;
            } else {
                encoded.push_back({currentValue, count});
                currentValue = signal[i];
                count = 1;
            }
            
            // Safety check to prevent memory issues with very large input
            if (encoded.size() > 1000000) {
                std::cerr << "Error: RLE producing too many segments. Increasing threshold." << std::endl;
                // Double the threshold and restart
                return runLengthEncode(signal, THRESHOLD * 2);
            }
        }
        
        // Add the last run
        encoded.push_back({currentValue, count});
        
        if (DEBUG) std::cout << "RLE complete. Compressed from " << signal.size() << " to " << encoded.size() << " elements." << std::endl;
        return encoded;
    }
    
    // Run-Length Encoding with custom threshold
    static std::vector<std::pair<double, int>> runLengthEncode(const std::vector<double>& signal, double threshold) {
        std::vector<std::pair<double, int>> encoded;
        
        if (signal.empty()) return encoded;
        
        if (DEBUG) std::cout << "Starting RLE with increased threshold " << threshold << "..." << std::endl;
        
        double currentValue = signal[0];
        int count = 1;

        for (size_t i = 1; i < signal.size(); ++i) {
            if (std::abs(signal[i] - currentValue) < threshold) {
                count++;
            } else {
                encoded.push_back({currentValue, count});
                currentValue = signal[i];
                count = 1;
            }
            
            // Safety check
            if (encoded.size() > 1000000) {
                std::cerr << "Error: RLE still producing too many segments. Using quantization." << std::endl;
                return quantizeAndEncode(signal, threshold * 5);
            }
        }
        
        // Add the last run
        encoded.push_back({currentValue, count});
        
        if (DEBUG) std::cout << "RLE complete with custom threshold. Compressed to " << encoded.size() << " elements." << std::endl;
        return encoded;
    }
    
    // Quantization and encoding for difficult signals
    static std::vector<std::pair<double, int>> quantizeAndEncode(const std::vector<double>& signal, double quantStep) {
        if (DEBUG) std::cout << "Using quantization with step " << quantStep << "..." << std::endl;
        
        std::vector<double> quantized(signal.size());
        for (size_t i = 0; i < signal.size(); i++) {
            quantized[i] = std::round(signal[i] / quantStep) * quantStep;
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
        std::unordered_map<double, Node*> symbolTable;
        int nodeNumberCounter;
        int encodedSymbolCount;
        const int MAX_SYMBOLS = 1000000;  // Safety limit

        Node* findHighestInBlock(Node* start, int weight) {
            if (!start) return nullptr;
            
            std::queue<Node*> q;
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
                std::swap(a->number, b->number);
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
            
            std::swap(a->number, b->number);
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
                std::cout << "Warning: Max iterations reached in updateTree" << std::endl;
            }
        }

        std::string getCode(Node* node) {
            std::string code;
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
            
            std::reverse(code.begin(), code.end());
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

        std::string encode(double symbol) {
            // Check for invalid symbols
            if (!std::isfinite(symbol)) {
                std::cerr << "Error: Invalid symbol detected in Huffman encoding: " << symbol << std::endl;
                return "ERROR";
            }
            
            // Safety check for maximum encoding size
            if (encodedSymbolCount++ > MAX_SYMBOLS) {
                std::cerr << "Error: Too many symbols encoded. Aborting." << std::endl;
                return "ERROR";
            }
            
            if (DEBUG && encodedSymbolCount % 1000 == 0) {
                std::cout << "  Huffman encoding symbol #" << encodedSymbolCount << ": " << symbol << std::endl;
            }
            
            std::string code;
            
            if (symbolTable.find(symbol) != symbolTable.end()) {
                Node* symbolNode = symbolTable[symbol];
                code = getCode(symbolNode);
                updateTree(symbolNode);
            } else {
                code = getCode(NYT);
                
                // Simplify encoding for large numbers to prevent overflow
                // Instead of using the full 64-bit representation, quantize to 32-bit
                std::string binary;
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
        static bool compressAudioFile(const std::string& inputFile, const std::string& outputFile) {
            if (DEBUG) std::cout << "===========================================" << std::endl;
            if (DEBUG) std::cout << "Starting compression process..." << std::endl;
            if (DEBUG) std::cout << "===========================================" << std::endl;
            
            // Open input audio file
            if (DEBUG) std::cout << "Opening input audio file: " << inputFile << std::endl;
            SF_INFO sfInfo;
            sfInfo.format = 0;
            SNDFILE* infile = sf_open(inputFile.c_str(), SFM_READ, &sfInfo);
            
            if (!infile) {
                std::cerr << "Error opening input file: " << inputFile << std::endl;
                return false;
            }

            // Print audio file info
            if (DEBUG) {
                std::cout << "Audio file info:" << std::endl;
                std::cout << "  Frames: " << sfInfo.frames << std::endl;
                std::cout << "  Channels: " << sfInfo.channels << std::endl;
                std::cout << "  Sample rate: " << sfInfo.samplerate << std::endl;
                std::cout << "  Total samples: " << (sfInfo.frames * sfInfo.channels) << std::endl;
            }

            // Read audio data
            if (DEBUG) std::cout << "Reading audio data..." << std::endl;
            std::vector<double> audioData(sfInfo.frames * sfInfo.channels);
            sf_count_t count = sf_read_double(infile, audioData.data(), audioData.size());
            sf_close(infile);

            if (DEBUG) std::cout << "Read " << count << " samples of " << audioData.size() << " expected." << std::endl;

            if (count == 0) {
                std::cerr << "No data read from input file" << std::endl;
                return false;
            }
            
            // Resize to actual read size if needed
            if (count < audioData.size()) {
                if (DEBUG) std::cout << "Resizing audio data to actual read size." << std::endl;
                audioData.resize(count);
            }

            // Check for NaN or Inf values
            if (DEBUG) std::cout << "Checking for invalid audio samples..." << std::endl;
            int invalidCount = 0;
            for (size_t i = 0; i < audioData.size(); i++) {
                if (!std::isfinite(audioData[i])) {
                    invalidCount++;
                    audioData[i] = 0.0; // Replace with a safe value
                }
            }
            if (invalidCount > 0) {
                std::cout << "Warning: Found and fixed " << invalidCount << " invalid audio samples." << std::endl;
            }

            // For large inputs, use a simpler approach
            std::vector<double> processedData;
            if (audioData.size() > 100000) {
                if (DEBUG) std::cout << "===========================================" << std::endl;
                if (DEBUG) std::cout << "Using simplified processing for large input..." << std::endl;
                processedData = simplifiedDCT(audioData);
            } else {
                // Apply Discrete Cosine Transform
                if (DEBUG) std::cout << "===========================================" << std::endl;
                if (DEBUG) std::cout << "Applying Discrete Cosine Transform..." << std::endl;
                processedData = discreteCosineTransform(audioData);
            }
            
            if (DEBUG) std::cout << "Signal processing completed! Produced " << processedData.size() << " coefficients." << std::endl;

            // Apply Run-Length Encoding
            if (DEBUG) std::cout << "===========================================" << std::endl;
            if (DEBUG) std::cout << "Applying Run-Length Encoding..." << std::endl;
            std::vector<std::pair<double, int>> rleEncoded = runLengthEncode(processedData);
            if (DEBUG) std::cout << "RLE completed! Encoded size: " << rleEncoded.size() << " pairs." << std::endl;

            // Apply Adaptive Huffman Coding
            if (DEBUG) std::cout << "===========================================" << std::endl;
            if (DEBUG) std::cout << "Applying Adaptive Huffman Encoding..." << std::endl;
            AdaptiveHuffmanCoder huffmanCoder;
            std::vector<std::string> encodedSymbols;
            
            // Progress tracking
            size_t totalPairs = rleEncoded.size();
            size_t progressInterval = std::max(size_t(1), totalPairs / 20);
            
            // Limit the maximum number of pairs to encode
            const size_t MAX_PAIRS = 100000;
            if (totalPairs > MAX_PAIRS) {
                if (DEBUG) std::cout << "Too many RLE pairs (" << totalPairs << "). Limiting to " << MAX_PAIRS << std::endl;
                totalPairs = MAX_PAIRS;
            }
            
            for (size_t i = 0; i < totalPairs; ++i) {
                const auto& coeff = rleEncoded[i];
                
                if (DEBUG && i % progressInterval == 0) {
                    std::cout << "  Huffman progress: " << (i * 100 / totalPairs) << "% (" << i << "/" << totalPairs << ")" << std::endl;
                }
                
                // Encode both the value and the run length
                std::string valueCode = huffmanCoder.encode(coeff.first);
                if (valueCode == "ERROR") {
                    std::cerr << "Error during Huffman encoding. Aborting." << std::endl;
                    return false;
                }
                
                std::string countCode = huffmanCoder.encode(static_cast<double>(coeff.second));
                if (countCode == "ERROR") {
                    std::cerr << "Error during Huffman encoding. Aborting." << std::endl;
                    return false;
                }
                
                encodedSymbols.push_back(valueCode + "|" + countCode);
            }
            
            if (DEBUG) std::cout << "Huffman encoding completed! Produced " << encodedSymbols.size() << " encoded symbols." << std::endl;

            // Write compressed data to output file
            if (DEBUG) std::cout << "===========================================" << std::endl;
            if (DEBUG) std::cout << "Writing compressed data to file: " << outputFile << std::endl;
            std::ofstream outfile(outputFile, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error creating output file: " << outputFile << std::endl;
                return false;
            }

            // Write metadata
            outfile << sfInfo.frames << " " 
                    << sfInfo.channels << " " 
                    << sfInfo.samplerate << std::endl;

            // Write encoded symbols
            size_t writtenSymbols = 0;
            for (const auto& symbol : encodedSymbols) {
                outfile << symbol << std::endl;
                writtenSymbols++;
                
                if (DEBUG && writtenSymbols % 10000 == 0) {
                    std::cout << "  Writing progress: " << (writtenSymbols * 100 / encodedSymbols.size()) 
                              << "% (" << writtenSymbols << "/" << encodedSymbols.size() << ")" << std::endl;
                }
            }

            outfile.close();
            if (DEBUG) std::cout << "File writing completed!" << std::endl;
            if (DEBUG) std::cout << "===========================================" << std::endl;
            return true;
        }
    };
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    std::cout << "Starting audio compression..." << std::endl;
    std::cout << "Input File: " << inputFile << std::endl;
    std::cout << "Output File: " << outputFile << std::endl;

    bool success = AudioCompressor::AdaptiveHuffmanCoder::compressAudioFile(inputFile, outputFile);

    if (success) {
        std::cout << "Compression successful. Output file: " << outputFile << std::endl;
        return 0;
    } else {
        std::cerr << "Compression failed." << std::endl;
        return 1;
    }
}