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



void Kohonen::initializeWeights() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);
  
  for (int i = 0; i < gridX_; i++) {
    for (int j = 0; j < gridY_; j++) {
      for (int k = 0; k < gridZ_; k++) {
        for (int l = 0; l < inputSize_; l++) {
          weights_[i][j][k][l] = dis(gen);
        }
      }
    }
  }
}

std::tuple<int, int, int> Kohonen::findBestMatchingUnit(const std::vector<double>& input) const {
  if (input.size() != static_cast<size_t>(inputSize_)) {
    throw std::runtime_error("Input size mismatch");
  }
  
  double minDist = std::numeric_limits<double>::max();
  std::tuple<int, int, int> bmu(-1, -1, -1);
  
  for (int i = 0; i < gridX_; i++) {
    for (int j = 0; j < gridY_; j++) {
      for (int k = 0; k < gridZ_; k++) {
        double dist = euclideanDistance(input, weights_[i][j][k]);
        if (dist < minDist) {
          minDist = dist;
          bmu = {i, j, k};
        }
      }
    }
  }
  return bmu;
}

double Kohonen::euclideanDistance(const std::vector<double>& input, const std::vector<double>& weight) const {
  double sum = 0.0;
  for (size_t i = 0; i < input.size(); i++) {
    double diff = input[i] - weight[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}



double Kohonen::neighborhoodFunction(double distance, double sigma) const {
  return std::exp(-distance * distance / (2 * sigma * sigma));
}

void Kohonen::updateWeights(const std::vector<double>& input, std::tuple<int, int, int> bmu, double learningRate, double sigma) {
  int bmuX = std::get<0>(bmu), bmuY = std::get<1>(bmu), bmuZ = std::get<2>(bmu);
  for (int i = 0; i < gridX_; i++) {
    for (int j = 0; j < gridY_; j++) {
      for (int k = 0; k < gridZ_; k++) {
        double distance = std::sqrt(std::pow(i - bmuX, 2) + std::pow(j - bmuY, 2) + std::pow(k - bmuZ, 2));
        if (distance <= sigma) {
          double influence = neighborhoodFunction(distance, sigma);
          for (int l = 0; l < inputSize_; l++) {
            weights_[i][j][k][l] += learningRate * influence * (input[l] - weights_[i][j][k][l]);
          }
        }
      }
    }
  }
}

void Kohonen::train() {
  std::cout << "Training with " << trainingData_.size() << " images\n";
  if (trainingData_.empty()) {
    std::cerr << "Error: No training data loaded" << std::endl;
    return;
  }
  
  for (int epoch = 0; epoch < epochs_; epoch++) {
    double learningRate = initialLearningRate_ * std::exp(-epoch / static_cast<double>(epochs_));
    double sigma = initialSigma_ * std::exp(-epoch / static_cast<double>(epochs_));
    
    for (const auto& image : trainingData_) {
      auto bmu = findBestMatchingUnit(image.pixels);
      updateWeights(image.pixels, bmu, learningRate, sigma);
    }
    double percentage = ((epoch + 1) * 100.0) / epochs_;
    std::cout << "Estamos en la época " << (epoch + 1) << " de " << epochs_ 
    << " con " << trainingData_.size() << " datos (" << percentage << "% completado)" << std::endl;
    std::cout.flush();
  }
}



void Kohonen::trainWithBatches(int batchSize) {
  std::cout << "Training with " << trainingData_.size() << " images in batches of " << batchSize << "\n";
  if (trainingData_.empty()) {
    std::cerr << "Error: No training data loaded" << std::endl;
    return;
  }
  
  for (int epoch = 0; epoch < epochs_; epoch++) {
    double learningRate = initialLearningRate_ * std::exp(-epoch / static_cast<double>(epochs_));
    double sigma = initialSigma_ * std::exp(-epoch / static_cast<double>(epochs_));
    
    for (size_t i = 0; i < trainingData_.size(); i += batchSize) {
      size_t end = std::min(i + batchSize, trainingData_.size());
      for (size_t j = i; j < end; j++) {
        auto bmu = findBestMatchingUnit(trainingData_[j].pixels);
        updateWeights(trainingData_[j].pixels, bmu, learningRate, sigma);
      }
      double epochPercentage = ((epoch + 1) * 100.0) / epochs_;
      double batchPercentage = ((i / static_cast<double>(batchSize) + 1) * 100.0) / (trainingData_.size() / batchSize);
      double totalPercentage = epochPercentage + (batchPercentage / epochs_);
      std::cout << "Estamos en la época " << (epoch + 1) << " de " << epochs_ 
      << " con " << trainingData_.size() << " datos (batch " << (i / batchSize + 1) 
      << " de " << (trainingData_.size() / batchSize) << ", " << totalPercentage << "% completado)" << std::endl;
      std::cout.flush();
    }
  }
}

void Kohonen::saveWeightsForVisualization(const std::string& outputFile) const {
  std::ofstream outFile(outputFile);
  if (!outFile.is_open()) {
    std::cerr << "Error: Could not open output file " << outputFile << std::endl;
    return;
  }
  
  // Asignar etiquetas a neuronas usando los datos de entrenamiento
  std::vector<std::vector<std::vector<int>>> neuronLabels(gridX_, std::vector<std::vector<int>>(gridY_, std::vector<int>(gridZ_, -1)));
  std::vector<std::vector<std::vector<int>>> labelCounts(gridX_, std::vector<std::vector<int>>(gridY_, std::vector<int>(gridZ_, 0)));
  std::ifstream trainFile("fashion-mnist_train.csv");
  std::string line;
  
  if (trainFile.is_open()) {
    getline(trainFile, line); // Ignorar encabezado
    while (getline(trainFile, line)) {
      std::stringstream ss(line);
      std::string token;
      int label;
      std::vector<double> pixels(inputSize_);
      getline(ss, token, ',');
      label = std::stoi(token);
      for (int i = 0; i < inputSize_; i++) {
        getline(ss, token, ',');
        pixels[i] = std::stod(token) / 255.0;
      }
      auto bmu = findBestMatchingUnit(pixels);
      int bmuX = std::get<0>(bmu), bmuY = std::get<1>(bmu), bmuZ = std::get<2>(bmu);
      neuronLabels[bmuX][bmuY][bmuZ] = label;
      labelCounts[bmuX][bmuY][bmuZ]++;
    }
    trainFile.close();
  }
  
  // Resolver conflictos con mayoría de votos (simplificado para 3D)
  for (int i = 0; i < gridX_; i++) {
    for (int j = 0; j < gridY_; j++) {
      for (int k = 0; k < gridZ_; k++) {
        if (labelCounts[i][j][k] > 0) {
          std::map<int, int> voteCount;
          for (int x = 0; x < gridX_; x++) {
            for (int y = 0; y < gridY_; y++) {
              for (int z = 0; z < gridZ_; z++) {
                if (neuronLabels[x][y][z] != -1) {
                  double dist = std::sqrt(std::pow(i - x, 2) + std::pow(j - y, 2) + std::pow(k - z, 2));
                  if (dist < 1.0) {
                    voteCount[neuronLabels[x][y][z]]++;
                  }
                }
              }
            }
          }
          int maxVote = -1, maxCount = 0;
          for (const auto& pair : voteCount) {
            if (pair.second > maxCount) {
              maxVote = pair.first;
              maxCount = pair.second;
            }
          }
          neuronLabels[i][j][k] = maxVote;
        }
      }
    }
  }
  
  // Guardar coordenadas y etiquetas con z derivado de pesos
  for (int i = 0; i < gridX_; i++) {
    for (int j = 0; j < gridY_; j++) {
      for (int k = 0; k < gridZ_; k++) {
        double z = std::accumulate(weights_[i][j][k].begin(), weights_[i][j][k].end(), 0.0) / inputSize_;
        outFile << i << "," << j << "," << k << "," << neuronLabels[i][j][k] << ",";
        for (int l = 0; l < inputSize_; l++) {
          outFile << weights_[i][j][k][l] << (l < inputSize_ - 1 ? "," : "\n");
        }
      }
    }
  }
  outFile.close();
  std::cout << "Weights and labels saved to " << outputFile << " for 3D visualization" << std::endl;
}



bool Kohonen::validateData(const std::string& filename) const {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return false;
  }
  
  std::string line;
  getline(file, line); // Ignorar encabezado
  int validLines = 0;
  
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string token;
    int pixelCount = 0;
    
    while (getline(ss, token, ',') && pixelCount < inputSize_) {
      try {
        std::stod(token);
        pixelCount++;
      } catch (const std::exception& e) {
        std::cerr << "Error: Invalid data in line" << std::endl;
        file.close();
        return false;
      }
    }
    if (pixelCount == inputSize_ + 1) validLines++;
  }
  file.close();
  std::cout << "Validated " << validLines << " images from " << filename << std::endl;
  return true;
}