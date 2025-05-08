#include <vector>
#include <map>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <cstdint>
#include <functional>

using namespace std;

//=====================================================================
// ALGORITHM 1: DCT (Discrete Cosine Transform)
//=====================================================================
// The DCT transforms time-domain audio samples into frequency domain,
// which enables more efficient compression through quantization.
//=====================================================================
namespace DCT {
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

    vector<double> apply(const vector<int16_t>& signal, int start, int length, const vector<vector<double>>& dctMatrix) {
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

    vector<int16_t> inverse(const vector<double>& transformed, const vector<vector<double>>& dctMatrix) {
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
}

//=====================================================================
// ALGORITHM 2: RLE (Run Length Encoding)
//=====================================================================
// RLE compresses data by storing sequences of repeated values as pairs 
// of (value, count). This is effective after DCT and quantization where 
// many consecutive zeros appear.
//=====================================================================
namespace RLE {
    vector<pair<int16_t, uint16_t>> encode(const vector<int16_t>& data) {
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

    vector<int16_t> decode(const vector<pair<int16_t, uint16_t>>& encoded) {
        vector<int16_t> decoded;
        for (const auto& p : encoded) {
            decoded.insert(decoded.end(), p.second, p.first);
        }
        return decoded;
    }
}

//=====================================================================
// ALGORITHM 3: Adaptive Huffman Compression
//=====================================================================
// Huffman coding is a variable-length prefix code that assigns shorter
// codes to more frequent symbols. The adaptive version updates the coding
// tree dynamically as data is processed without requiring a predefined
// frequency table.
//=====================================================================
namespace Huffman {
    struct HuffmanNode {
        int16_t symbol;
        int weight, order;
        HuffmanNode *parent, *left, *right;

        HuffmanNode(int16_t sym, int wt, int ord) 
            : symbol(sym), weight(wt), order(ord), parent(nullptr), left(nullptr), right(nullptr) {}
    };

    class AdaptiveHuffman {
    private:
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

    //-------------------------------------------------------------------
    // Bit Packing Utilities (part of Huffman compression)
    //-------------------------------------------------------------------
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
} 