#ifndef KOHONEN_H
#define KOHONEN_H

#include <vector>
#include <string>
#include <utility>
#include <limits>

class Kohonen {
public:
    // Constructor para cuadrícula 3D
    Kohonen(int gridX, int gridY, int gridZ, int inputSize, int epochs, double initialLearningRate, double initialSigma);

    // Métodos públicos
    bool loadData(const std::string& filename);
    void train();
    void trainWithBatches(int batchSize); // Entrenamiento por lotes
    void saveWeightsForVisualization(const std::string& outputFile) const;
    bool validateData(const std::string& filename) const;

    // Getters
    int getGridX() const { return gridX_; }
    int getGridY() const { return gridY_; }
    int getGridZ() const { return gridZ_; }
    const std::vector<std::vector<std::vector<std::vector<double>>>>& getWeights() const { return weights_; }

private:
    // Estructura para una imagen
    struct Image {
        std::vector<double> pixels;
        Image() : pixels() {}  // Constructor por defecto
        Image(const std::vector<double>& p) : pixels(p) {}  // Constructor con argumento
    };

    // Configuración
    const int gridX_, gridY_, gridZ_, inputSize_, epochs_;
    double initialLearningRate_, initialSigma_;

    // Datos y pesos
    std::vector<Image> trainingData_;
    std::vector<std::vector<std::vector<std::vector<double>>>> weights_; // 3D weights

    // Métodos privados
    void initializeWeights();
    std::tuple<int, int, int> findBestMatchingUnit(const std::vector<double>& input) const;
    double euclideanDistance(const std::vector<double>& input, const std::vector<double>& weight) const;
    double neighborhoodFunction(double distance, double sigma) const;
    void updateWeights(const std::vector<double>& input, std::tuple<int, int, int> bmu, double learningRate, double sigma);
};

#endif