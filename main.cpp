#include "kohonen.h"
#include <iostream>

int main() {
    try {
        std::cout << "Starting Kohonen network initialization...\n";
        // Configuración de la red Kohonen 3D 10x10x10
        Kohonen som(10, 10, 10, 784, 10, 0.1, 5.0); // Reducido initialSigma a 5.0 para cuadrícula más pequeña

        std::cout << "Loading training data...\n";
        if (!som.loadData("AfroTrain.csv")) {
            return 1;
        }

        std::cout << "Validating test data...\n";
        if (!som.validateData("AfroTest.csv")) {
            std::cerr << "Validation failed, proceeding with training anyway\n";
        }

        std::cout << "Starting training...\n";
        // Entrenar la red con batches
        som.trainWithBatches(100); // Usar lotes de 100 imágenes

        std::cout << "Saving weights for visualization...\n";
        som.saveWeightsForVisualization("som_output.txt");

        std::cout << "Process completed successfully.\n";
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}