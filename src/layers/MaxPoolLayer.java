package layers;

import java.util.List;

//https://youtu.be/8WrEz-M50oQ?list=PLpcNcOt2pg8k_YsrMjSwVdy3GX-rc_ZgN

public class MaxPoolLayer extends Layer {
    @Override
    public double[] getOutput(List<double[][]> input) {
        return new double[0];
    }

    @Override
    public double[] getOutput(double[] input) {
        return new double[0];
    }

    @Override
    public void backPropagation(double[] lossGradient) {

    }

    @Override
    public void backPropagation(List<double[][]> lossGradient) {

    }

    @Override
    public int getOutputLength() {
        return 0;
    }

    @Override
    public int getOutputRows() {
        return 0;
    }

    @Override
    public int getOutputColumns() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return 0;
    }
}
