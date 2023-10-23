package layers;

import java.util.List;

public class ConcolutionLayer extends Layer{
    @Override
    public double[] getOutput(List<double[][]> input) throws Exception {
        return new double[0];
    }

    @Override
    public double[] getOutput(double[] input) throws Exception {
        return new double[0];
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

    }

    @Override
    public void backPropagation(double[] dLdO) {

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
    public int getOutputCols() {
        return 0;
    }

    @Override
    public int getOutputElements() {
        return 0;
    }
}
