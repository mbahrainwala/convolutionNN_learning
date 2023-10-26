package layers;

import data.MatrixUtility;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer extends Layer{

    private final long seed;
    private List<double[][]> _filters;
    private List<double[][]> _lastInput;
    private final int _filterSize;
    private final int _stepSize;

    private final int _inLength;
    private final int _inRows;
    private final int _inCols;
    private final double learningRate;

    public ConvolutionLayer(int filterSize, int stepSize, int inLength, int inRows, int inCols, long seed, int numFilters, double learningRate) {
        this.seed = seed;
        _filterSize = filterSize;
        _stepSize = stepSize;
        _inLength = inLength;
        _inRows = inRows;
        _inCols = inCols;
        this.learningRate = learningRate;

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
        _lastInput = list;

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

    /**
     * <b>NOTE:</b><br/>
     * To calculate back proposition for a convolution layer we need to follow the chain rule<br/>
     *<br/>
     * Input Matrix<br/>
     * +---+---+---+---+---+<br/>
     * |X11|X12|X13|X14|X15|<br/>
     * +---+---+---+---+---+<br/>
     * |X21|X22|X23|X24|X25|<br/>
     * +---+---+---+---+---+<br/>
     * |X31|X32|X33|X34|X35|<br/>
     * +---+---+---+---+---+<br/>
     * |X41|X42|X43|X44|X45|<br/>
     * +---+---+---+---+---+<br/>
     * |X51|X52|X53|X54|X55|<br/>
     * +---+---+---+---+---+<br/>
     *<br/>
     * Filter matrix<br/>
     * +---+---+---+<br/>
     * |F11|F12|F13|<br/>
     * +---+---+---+<br/>
     * |F21|F22|F23|<br/>
     * +---+---+---+<br/>
     * |F31|F32|F33|<br/>
     * +---+---+---+<br/>
     *<br/>
     * Output matrix<br/>
     * +---+---+<br/>
     * |O11|O12|<br/>
     * +---+---+<br/>
     * |O21|O22|<br/>
     * +---+---+<br/>
     *<br/>
     *<br/>
     * dL/dF -> change in loss relative to change in filter value<br/>
     * dL/dO -> change in loss relative to change in output value<br/>
     * dO/dF -> change in output relative to change in filter value<br/>
     * dL/dF11 = dL/dO*dO/dF11 = (dL/dO11*dO11/dF11)+(dL/dO12*dO12/dF11)+(dL/dO21*dO21/dF11)+(dL/dO22*dO22/dF11)<br/>
     *<br/>
     * F11 -> filter value at List(double[][]) _filters->filter[1][1]<br/>
     *<br/>
     * When calculating O11 in the forward pass we do<br/>
     * O11 = X11*F11+S12*F12+X13*F13 + X21*F21+X22*F22+X23*F23 + X31*F31+X32*F32+X33*F33<br/>
     * depending on the step size we either move by 1 or 2 rows and the columns to get the next output matrix.<br/>
     * e.g. for step size=2<br/>
     * O12 = X13*F11+X14*F12+X15*F13 + X23*F21+X24*F25+X23*F23 + X31*F33+X34*F32+X35*F33<br/>
     *<br/>
     * O21 = X31*F11+X32*F32+X33*F13 + X41*F21+X42*F22+X43*F23 + X51*F31+X52*F32+X53*F33<br/>
     * O22 = X33*F11+X34*F12+X35*F13 + X43*F21+X34*F25+X33*F23 + X41*F33+X44*F32+X45*F33<br/>
     *<br/>
     * By differentiating we will get. This is how the math works out.<br/>
     * O11 -> dO11/dF11 = X11 = input[1][1]<br/>
     * O12 -> dO12/dF11 = X13 = input[1][3]<br/>
     * O21 -> dO21/dF11 = X31 = input[3][1]<br/>
     * O22 -> dO22/dF11 = X33 = input[3][3]<br/>
     *<br/>
     * Loss for filter[1][1] = dL/dF11 = dL/dO11*X11 + dl/dO12*X13 + dl/dO21*X31 + dL/dO22*X33<br/>
     * Loss for filter[1][2] = dL/dF12 = dL/dO11*X12 + dl/dO12*X14 + dl/dO21*X32 + dL/dO22*X34<br/>
     * ...<br/>
     */

    @Override
    public void backPropagation(List<double[][]> dLdO) {

        List<double[][]> filtersDelta = new ArrayList<>();
        List<double[][]> dLdOPreviousLayer = new ArrayList<>();

        for(int f=0; f< _filters.size(); f++){
            filtersDelta.add(new double[_filterSize][_filterSize]);
        }

        for(int i=0; i<_lastInput.size(); i++){
            double[][] errorForInput = new double[_inRows][_inCols];
            for(int f=0; f< _filters.size(); f++){
                double[][] currFilter = _filters.get(f);
                double[][] error = dLdO.get(i*_filters.size()+f);

                double[][] spacedError = spaceArray(error);
                double[][] dLdF= convolve(_lastInput.get(i), spacedError, 1);

                double[][] delta = MatrixUtility.multiply(dLdF, learningRate*-1);
                double[][] newDelta = MatrixUtility.add(filtersDelta.get(f), delta);
                filtersDelta.set(f, newDelta);

                if(get_previousLayer()!=null) {
                    double[][] flippedError = MatrixUtility.flipMatrixHorizontal(MatrixUtility.flipMatrixVertical(spacedError));
                    errorForInput = MatrixUtility.add(errorForInput, fullConvolve(currFilter, flippedError));
                }
            }

            if(get_previousLayer()!=null)
                dLdOPreviousLayer.add(errorForInput);
        }

        for(int f=0; f< _filters.size(); f++){
            double[][] modified = MatrixUtility.add(filtersDelta.get(f), _filters.get(f));
            _filters.set(f, modified);
        }

        if(get_previousLayer()!=null)
        {
            get_previousLayer().backPropagation(dLdOPreviousLayer);
        }
    }

    @Override
    public void backPropagation(double[] dLdO) {
        backPropagation(vectorToMatrix(dLdO, _inLength, _inRows, _inCols));
    }

    /**
     * This will create a spaced out matrix based on our input.
     * @param input
     * @return
     */
    private double[][] spaceArray(double[][] input){
        if(_stepSize < 2){
            return input;
        }

        int outRows = (input.length - 1)* _stepSize + 1;
        int outCols = (input[0].length -1)*_stepSize+1;

        double[][] output = new double[outRows][outCols];

        for(int i = 0; i < input.length; i++){
            for(int j = 0; j < input[0].length; j++){
                output[i*_stepSize][j*_stepSize] = input[i][j];
            }
        }

        return output;
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

    private double[][] fullConvolve(double[][] input, double[][] filter){
        int outRows = (input.length + filter.length)+1;
        int outCols = (input[0].length + filter[0].length)+1;

        int inRows = input.length;
        int inCols = input[0].length;

        int fRows = filter.length;
        int fCols = filter[0].length;

        double[][] output = new double[outRows][outCols];

        int outRow=0;

        for(int i= -fRows + 1; i> inRows - fRows; i++){
            int outCol=0;
            for(int j= -fCols+1; j> inCols - fCols; j++){
                double sum = 0;
                //Apply the filter around this position
                for(int x=0; x<fRows; x++){
                    for(int y=0; y<fCols; y++){
                        int inputRowIndex = i+x;
                        int inputColIndex = j+y;

                        if(inputRowIndex >=0 && inputColIndex >=0 && inputRowIndex < inRows && inputColIndex < inCols){
                            double value = filter[x][y]*input[inputRowIndex][inputColIndex];
                            sum+=value;
                        }
                    }
                }
                output[outRow][outCol] = sum;
                outCol++;
            }
            outRow++;
        }

        return output;
    }
}
