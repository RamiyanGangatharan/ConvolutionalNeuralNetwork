package layers;

import java.util.ArrayList;
import java.util.List;

/**
 * Abstract base class representing a single layer in a neural network.
 *
 * Each layer stores references to the previous and next layers, handles forward
 * propagation to compute outputs, and defines abstract methods for backpropagation
 * and output dimensions. This class also provides utility methods to convert
 * between 3D matrices (List of 2D arrays) and flattened 1D vectors, which is
 * often needed in neural network computations.
 */
public abstract class Layer {

    // References to neighboring layers in the network
    protected Layer nextLayer;
    protected Layer prevLayer;

    public Layer getNextLayer() { return nextLayer; }
    public Layer getPrevLayer() { return prevLayer; }

    public void setNextLayer(Layer nextLayer) { this.nextLayer = nextLayer; }
    public void setPrevLayer(Layer prevLayer) { this.prevLayer = prevLayer; }

    /**
     * Forward propagation using a matrix-style input (e.g., image).
     *
     * @param input a list of 2D double arrays representing input data
     * @return the output of this layer as a flattened 1D array
     */
    public abstract double[] getOutput(List<double[][]> input);

    /**
     * Forward propagation using a vector-style input.
     *
     * @param input a 1D array representing the input to this layer
     * @return the output of this layer as a 1D array
     */
    public abstract double[] getOutput(double[] input);

    // Backpropagation uses the derivative of the loss function with respect to the output

    /**
     * Performs backpropagation for vector-style input.
     *
     * @param lossGradient the derivative of the loss function with respect to this layer's output (dL/dO)
     */
    public abstract void backPropagation(double[] lossGradient);

    /**
     * Performs backpropagation for matrix-style input.
     *
     * @param lossGradient the derivative of the loss function with respect to this layer's output as a list of 2D arrays
     */
    public abstract void backPropagation(List<double[][]> lossGradient);

    public abstract int getOutputLength();
    public abstract int getOutputRows();
    public abstract int getOutputColumns();
    public abstract int getOutputElements();

    // ---------------------------
    // Conversion utilities
    // ---------------------------

    /**
     * Flattens a list of 2D matrices into a single 1D vector.
     *
     * @param input list of 2D double arrays (e.g., multiple images)
     * @return flattened 1D vector containing all elements
     */
    public double[] matrixToVector(List<double[][]> input) {
        int length = input.size();
        int rows = input.get(0).length;
        int cols = input.get(0)[0].length;

        double[] vector = new double[length * rows * cols];

        int i = 0;

        for (int l = 0; l < length; l++) {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    vector[i] = input.get(l)[r][c];
                    i++;
                }
            }
        }
        return vector;
    }

    /**
     * Converts a flattened vector into a list of 2D matrices.
     *
     * @param input flattened 1D array
     * @param length number of matrices to create
     * @param rows number of rows per matrix
     * @param columns number of columns per matrix
     * @return list of reconstructed 2D double arrays
     */
    public List<double[][]> vectorToMatrix(double[] input, int length, int rows, int columns) {
        List<double[][]> output = new ArrayList<>();

        int i = 0;

        for (int l = 0; l < length; l++) {
            double [][] matrix = new double[rows][columns];

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < columns; c++) {
                    matrix[r][c] = input[i];
                    i++;
                }
            }
            output.add(matrix);
        }
        return output;
    }
}
