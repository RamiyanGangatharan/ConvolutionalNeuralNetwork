package layers;

import java.util.List;
import java.util.Random;

/**
 * Represents a fully connected (dense) neural network layer.
 * <a href="https://www.youtube.com/watch?v=JJUlkPFq1q8&list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN&index=4">Tutorial Segment</a>
 *
 * <p>This layer performs two core operations:</p>
 *
 * <h3>1. Forward Pass</h3>
 * <ul>
 *   <li>Computes pre-activation values Z using the weighted sum: Z = W·X</li>
 *   <li>Applies the ReLU activation function to produce the output</li>
 *   <li>Stores the input (X) and pre-activation values (Z) for use during backpropagation</li>
 * </ul>
 *
 * <h3>2. Backpropagation</h3>
 * <p>The backward pass computes gradients for three things:</p>
 * <ul>
 *   <li><b>The gradient of the loss with respect to each weight (dL/dW)</b>,
 *       which determines how the weights should be updated</li>
 *   <li><b>The gradient of the loss with respect to each input (dL/dX)</b>,
 *       which is passed to the previous layer</li>
 *   <li><b>The ReLU derivative (dO/dZ)</b>, which gates gradients based on
 *       whether each neuron was active during the forward pass</li>
 * </ul>
 *
 * <p>Weight updates follow standard gradient descent:
 * <pre>
 *     W := W - learningRate * gradient
 * </pre>
 * </p>
 *
 * <h3>Stored Values</h3>
 * <ul>
 *   <li><b>lastX</b>: the input from the forward pass</li>
 *   <li><b>lastZ</b>: the pre-activation values before ReLU</li>
 *   <li><b>weights</b>: matrix of trainable parameters</li>
 *   <li><b>learningRate</b>: update rate used during backpropagation</li>
 * </ul>
 *
 * <h3>Usage</h3>
 * <p>This layer can be used in a chain of layers. Its output is passed to the next layer
 * during the forward pass, and its input gradient is sent to the previous layer during backprop.</p>
 *
 * <p>Supports both vector and matrix inputs via the parent {@link Layer} class.</p>
 */
public class FullyConnectedLayer extends Layer {

    private long SEED;
    private final double leak = 0.01;

    private double[][] weights;
    private int inputLength;
    private int outputLength;
    private double learningRate;

    private double[] lastZ;
    private double[] lastX;

    /**
     * Creates a fully connected layer with randomly initialized weights.
     *
     * @param inputLength number of input neurons
     * @param outputLength number of output neurons
     * @param SEED RNG seed used for weight initialization
     * @param learningRate gradient descent learning rate
     */
    public FullyConnectedLayer(int inputLength, int outputLength, long SEED, double learningRate) {
        this.inputLength = inputLength;
        this.outputLength = outputLength;
        this.SEED = SEED;
        this.learningRate = learningRate;

        weights = new double[inputLength][outputLength];
        setRandomWeights();
    }

    /**
     * Performs the forward pass for a vector input.
     *
     * <p>Computes the weighted sum for each output neuron and then applies
     * ReLU activation. The input and pre-activation values are stored for
     * use during backpropagation.</p>
     *
     * @param input input vector
     * @return output vector after ReLU activation
     */
    public double[] fullyConnectedForwardPass(double[] input) {

        lastX = input;

        double [] z = new double[outputLength];
        double [] out = new double[outputLength];

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                z[j] += input[i] * weights[i][j];
            }
        }

        lastZ = z;

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                out[j] = RELU(z[j]);
            }
        }
        return out;
    }

    @Override public double[] getOutput(List<double[][]> input) {
        double[] vector = matrixToVector(input);
        return getOutput(vector);
    }

    @Override public double[] getOutput(double[] input) {
        double[] forwardPass = fullyConnectedForwardPass(input);
        if (nextLayer != null) { return nextLayer.getOutput(forwardPass); }
        else { return forwardPass; }
    }

    /**
     * Performs backpropagation on this layer, updating weights and computing
     * the gradient to pass backward to the previous layer.
     *
     * @param lossGradient gradient of the loss with respect to this layer's output (dL/dO)
     */
    @Override public void backPropagation(double[] lossGradient) {

        double[] lossGradientWRTInput = new double[inputLength]; // dLdx

        // NOTE: WRT means with respect to. AND dLDO is loss gradient.
        // derivative of the output with respect to Z (pre-activation values)
        double outputWRTPreActivation; // [d0dz]

        // derivative of the pre-activation value with respect to each weight
        double preActivationWRTWeight; // [dzdw]

        // derivative of the loss with respect to each weight
        double lossWRTWeight; // [dLdw]

        for (int k = 0; k < inputLength; k++) {
            double lossGradientSummation = 0;

            for (int j = 0; j < outputLength; j++) {
                outputWRTPreActivation = RELU_DERIVATIVE(lastZ[j]);
                preActivationWRTWeight = lastX[k];
                double preActivationWRTInput = weights[k][j];

                lossWRTWeight = lossGradient[j] * outputWRTPreActivation * preActivationWRTWeight;
                weights[k][j] -= lossWRTWeight * learningRate;
                lossGradientSummation += lossGradient[j] * outputWRTPreActivation * preActivationWRTInput;
            }
            lossGradientWRTInput[k] = lossGradientSummation;
        }
        if (prevLayer != null) { prevLayer.backPropagation(lossGradientWRTInput); }
    }

    @Override public void backPropagation(List<double[][]> lossGradient) {
        double[] vector = matrixToVector(lossGradient);
        backPropagation(vector);
    }

    @Override public int getOutputLength() {
        return 0;
    }

    @Override public int getOutputRows() {
        return 0;
    }

    @Override public int getOutputColumns() {
        return 0;
    }

    /**
     * Returns the number of elements in this layer's output vector.
     *
     * @return output vector length
     */
    @Override public int getOutputElements() {
        return outputLength;
    }

    /**
     * Initializes all weights to values drawn from a Gaussian distribution.
     */
    public void setRandomWeights() {
        Random rand = new Random(SEED);

        for (int i = 0; i < inputLength; i++) {
            for  (int j = 0; j < outputLength; j++) {
                weights[i][j] = rand.nextGaussian();
            }
        }
    }

    // RELU Activation Functions
    // NOTE: RELU means Rectified Linear Unit (ReLU)

    /**
     * Applies the ReLU activation function.
     *
     * @param input value to activate
     * @return input if positive, otherwise 0
     */
    public double RELU(double input) {
        if (input <= 0) { return 0; }
        else { return input; }
    }


    /**
     * Computes the derivative of the ReLU activation function.
     *
     * @param input pre-activation value
     * @return 1 if input > 0, otherwise a small leak value
     */
    public double RELU_DERIVATIVE(double input) {
        if (input <= 0) { return leak; }
        else { return 1; }
    }
}
