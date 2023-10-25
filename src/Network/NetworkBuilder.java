package Network;

import layers.*;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {
    private NeuralNetwork nn;
    private final int _inputRows;
    private final int _inputCols;
    private final double _scaleFactor;
    private final List<Layer> _layers = new ArrayList<>();

    public NetworkBuilder(int inputRows, int inputCols, double scaleFactor) {
        _inputRows = inputRows;
        _inputCols = inputCols;
        _scaleFactor = scaleFactor;
    }

    public void addConvolutionLayer(int numFilters, int filterSize, int stepSize, double learningRate, long SEED){
        if(_layers.isEmpty())
            _layers.add(new ConvolutionLayer(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    public void addConvolutionLayer2(int numFilters, int filterSize, int stepSize, double learningRate, long SEED){
        if(_layers.isEmpty())
            _layers.add(new ConvolutionLayer2(filterSize, stepSize, 1, _inputRows, _inputCols, SEED, numFilters, learningRate));
        else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new ConvolutionLayer2(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), SEED, numFilters, learningRate));
        }
    }

    public void addMaxPoolLayer(int windowSize, int stepSize){
        if(_layers.isEmpty())
            _layers.add(new MaxPoolLayer(stepSize, windowSize, 1, _inputRows, _inputCols));
        else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addMaxPoolLayer2(int windowSize, int stepSize){
        if(_layers.isEmpty())
            _layers.add(new MaxPoolLayer2(stepSize, windowSize, 1, _inputRows, _inputCols));
        else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new MaxPoolLayer2(stepSize, windowSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols()));
        }
    }

    public void addFullyConnectedLayer(int outLength, double learningRate, long SEED){
        //public FullyConnectedLayer(int inLenght,int outLength, long seed, double learningRate)
        if(_layers.isEmpty())
            _layers.add(new FullyConnectedLayer( _inputCols*_inputRows, outLength, SEED, learningRate));
        else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new FullyConnectedLayer(prev.getOutputElements(), outLength, SEED, learningRate));
        }
    }

    public void addFullyConnectedLayer2(int outLength, double learningRate, long SEED){
        //public FullyConnectedLayer(int inLenght,int outLength, long seed, double learningRate)
        if(_layers.isEmpty())
            _layers.add(new FullyConnectedLayer2( _inputCols*_inputRows, outLength, SEED, learningRate));
        else{
            Layer prev = _layers.get(_layers.size()-1);
            _layers.add(new FullyConnectedLayer2(prev.getOutputElements(), outLength, SEED, learningRate));
        }
    }

    public NeuralNetwork build(){
        nn = new NeuralNetwork(_layers, _scaleFactor);
        return nn;
    }
}
