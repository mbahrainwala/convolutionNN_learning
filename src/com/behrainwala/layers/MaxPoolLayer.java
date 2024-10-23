package com.behrainwala.layers;

import java.util.ArrayList;
import java.util.List;

public class MaxPoolLayer extends Layer{

    private final int _stepSize;
    private final int _windowSize;
    private final int _inLength;
    private final int inRows;
    private final int inCols;
    private List<int[][]> _lastMaxRow;
    private List<int[][]> _lastMaxCol;

    public MaxPoolLayer(int _stepSize, int _windowSize, int _inLength, int inRows, int inCols) {
        this._stepSize = _stepSize;
        this._windowSize = _windowSize;
        this._inLength = _inLength;
        this.inRows = inRows;
        this.inCols = inCols;
    }

    private List<double[][]> maxPoolForwardPass(List<double[][]> input){
        List<double[][]> output = new ArrayList<>();
        _lastMaxRow = new ArrayList<>();
        _lastMaxCol = new ArrayList<>();


        for (double[][] doubles : input) {
            output.add(pool(doubles));
        }

        return output;
    }

    private double[][] pool(double[][] input){
        double[][] output = new double[getOutputRows()][getOutputCols()];

        int[][] maxRows = new int[getOutputRows()][getOutputCols()];
        int[][] maxCols = new int[getOutputRows()][getOutputCols()];

        for(int r=0; r< getOutputRows(); r+=_stepSize){
            for(int c=0; c<getOutputCols(); c+=_stepSize){

                double max = 0.0;
                maxRows[r][c] = -1;
                maxCols[r][c] = -1;

                for(int x=0; x< _windowSize; x++){
                    for(int y=0; y< _windowSize; y++){
                        if(max < input[r+x][c+y]) {
                            max = input[r + x][c + y];
                            maxRows[r][c] = r+x;
                            maxCols[r][c] = c+y;
                        }
                    }
                }

                output[r][c]=max;
            }
        }

        _lastMaxRow.add(maxRows);
        _lastMaxCol.add(maxCols);
        return output;

    }

    @Override
    public double[] getOutput(List<double[][]> input) throws Exception {
        List<double[][]> outputPool = maxPoolForwardPass(input);

        if(get_nextLayer()!=null)
            return get_nextLayer().getOutput(outputPool);
        else
            throw new Exception("This cannot be the last layer");
    }

    @Override
    public double[] getOutput(double[] input) throws Exception {
        return getOutput(vectorToMatrix(input, _inLength, inRows, inCols));
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        List<double[][]>dX_dL = new ArrayList<>();

        int l=0;
        for(double[][] array:dLdO){
            double[][] error = new double[inRows][inCols];

            for(int r=0; r<getOutputRows(); r++){
                for(int c=0; c<getOutputCols(); c++){
                    int max_x = _lastMaxRow.get(l)[r][c];
                    int max_y = _lastMaxCol.get(l)[r][c];

                    if(max_x != -1){
                        error[max_x][max_y] += array[r][c];
                    }
                }
            }
            dX_dL.add(error);
            l++;
        }

        if(get_previousLayer()!=null)
            get_previousLayer().backPropagation(dX_dL);
    }

    @Override
    public void backPropagation(double[] dLdO) {
        backPropagation(vectorToMatrix(dLdO, getOutputLength(), getOutputRows(), getOutputCols()));
    }

    @Override
    public int getOutputLength() {
        return _inLength;
    }

    @Override
    public int getOutputRows() {
        return (inRows-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputCols() {
        return (inCols-_windowSize)/_stepSize + 1;
    }

    @Override
    public int getOutputElements() {
        return _inLength*getOutputCols()*getOutputRows();
    }
}
