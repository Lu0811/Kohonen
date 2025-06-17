#include "kohonen.h"
#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <sstream>
#include <map>
#include <numeric>

Kohonen::Kohonen(int gridX, int gridY, int gridZ, int inputSize, int epochs, double initialLearningRate, double initialSigma)
    : gridX_(gridX), gridY_(gridY), gridZ_(gridZ), inputSize_(inputSize), epochs_(epochs),
      initialLearningRate_(initialLearningRate), initialSigma_(initialSigma) {
    if (gridX_ <= 0 || gridY_ <= 0 || gridZ_ <= 0 || inputSize_ <= 0 || epochs_ <= 0 ||
        initialLearningRate_ <= 0.0 || initialSigma_ <= 0.0) {
        throw std::invalid_argument("Invalid parameters for Kohonen network initialization");
    }
    weights_.resize(gridX_, std::vector<std::vector<std::vector<double>>>(gridY_, std::vector<std::vector<double>>(gridZ_, std::vector<double>(inputSize_))));
    initializeWeights();
}

bool Kohonen::loadData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << ". Check path and permissions." << std::endl;
        return false;
    }

    std::string line;
    if (getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> header;
        while (getline(ss, token, ',')) {
            header.push_back(token);
        }
        if (header[0] != "label" || header.size() != inputSize_ + 1) {
            std::cerr << "Error: Invalid header format. Expected 'label' followed by " << inputSize_ 
                      << " pixels, got " << header.size() - 1 << " pixels." << std::endl;
            file.close();
            return false;
        }
        std::cout << "Header verified successfully: " << line << std::endl;
    } else {
        std::cerr << "Error: Empty file or no header in " << filename << std::endl;
        file.close();
        return false;
    }

    trainingData_.clear();
    std::vector<int> labels;

    int lineCount = 0;
    while (getline(file, line)) {
        lineCount++;
        std::stringstream ss(line);
        std::string token;
        int label;
        if (!getline(ss, token, ',')) {
            std::cerr << "Error: Missing label in line " << lineCount << std::endl;
            file.close();
            return false;
        }
        label = std::stoi(token);
        labels.push_back(label);
        std::vector<double> pixels(inputSize_);
        for (int i = 0; i < inputSize_; i++) {
            if (!getline(ss, token, ',')) {
                std::cerr << "Error: Insufficient pixel data in line " << lineCount 
                          << ". Expected " << inputSize_ << " pixels, got " << i << std::endl;
                file.close();
                return false;
            }
            try {
                pixels[i] = std::stod(token) / 255.0;
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid pixel value in line " << lineCount << ": " << e.what() << std::endl;
                file.close();
                return false;
            }
        }
        trainingData_.emplace_back(pixels);
    }
    file.close();
    std::cout << "Loaded " << trainingData_.size() << " images with labels from " << filename << std::endl;
    return true;
}