package layers;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer{

    private final long seed;
    private List<double[][]> _filters;
    private final int _filterSize;
    private final int _stepSize;

    private final int _inLength;
    private final int _inRows;
    private final int _inCols;

    public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows, int inCols, long seed, int numFilters) {
        this.seed = seed;
        _filterSize = filterSize;
        _stepSize = stepSize;
        _inLength = inLength;
        _inRows = inRows;
        _inCols = inCols;

        generateFilters(numFilters);
    }

    private void generateFilters(int numFilters){
        List<double[][]> filters = new ArrayList<>(numFilters);
        Random rand = new Random(seed);

        for(int n=0; n<numFilters; n++){
            double[][] newFilter = new double[_filterSize][_filterSize];

            for(int i=0; i<_filterSize; i++){
                for(int j=0; j<_filterSize; j++){
                    newFilter[i][j] = rand.nextGaussian();
                }
            }

            filters.add(newFilter);
        }

        _filters = filters;
    }

    private List<double[][]> convolutionForwardPass(List<double[][]> list){
        List<double[][]> output = new ArrayList<>();

        for (double[][] input : list) {
            for (double[][] filter : _filters) {
                output.add(convolve(input, filter, _stepSize));
            }
        }

        return output;
    }

    private double[][] convolve(double[][] input, double[][] filter, int stepSize){
        int outRows = (input.length - filter.length)/stepSize+1;
        int outCols = (input[0].length - filter[0].length)/stepSize+1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow=0;

        for(int i=0; i<= inRows - fRows; i+=stepSize){
            int outCol=0;
            for(int j=0; j<= inCols - fCols; j+=stepSize){
                double sum = 0;
                //Apply the filter around this position
                for(int x=0; x<fRows; x++){
                    for(int y=0; y<fCols; y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        double value = filter[x][y]*input[inputRowIndex][inputColIndex];
                        sum+=value;
                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }

    @Override
    public double[] getOutput(List<double[][]> input) throws Exception {

        if(get_nextLayer()!=null)
            return get_nextLayer().getOutput(convolutionForwardPass(input));
        else
            throw new Exception("This cannot be the last layer");
    }

    @Override
    public double[] getOutput(double[] input) throws Exception {
        return getOutput(convolutionForwardPass(vectorToMatrix(input, _inLength, _inRows, _inCols)));
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {

    }

    @Override
    public void backPropagation(double[] dLdO) {

    }

    @Override
    public int getOutputLength() {
        return _filters.size()*_inLength;
    }

    @Override
    public int getOutputRows() {
        return (_inRows-_filterSize)/_stepSize+1;
    }

    @Override
    public int getOutputCols() {
        return (_inCols-_filterSize)/_stepSize+1;
    }

    @Override
    public int getOutputElements() {
        return getOutputRows()*getOutputCols()*getOutputLength();
    }
}
