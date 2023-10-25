package Network;

import data.Image;
import data.MatrixUtility;
import layers.Layer;

import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private final List<Layer> _layers;
    private final double scaleFactor;

    public NeuralNetwork(List<Layer> layers, double scaleFactor) {
        _layers = layers;
        this.scaleFactor = scaleFactor;
        linkLayers();
    }

    private void linkLayers(){
        if(_layers.size() < 2){
            return;
        }

        for(int i=0; i< _layers.size(); i++){
            if(i!= _layers.size()-1)
                _layers.get(i).set_nextLayer(_layers.get(i+1));
            if(i!=0)
                _layers.get(i).set_previousLayer(_layers.get(i-1));
        }
    }

    public double[] getErrors(double[] networkOutput, int correctAnswer){
        int numClasses = networkOutput.length;

        double[] expected = new double[numClasses];
        expected[correctAnswer] = 1;
        /**
         * all the expected output vectors should be 0 except for the correct one which should be 1.
         *
         * subtract the expected from the network output to get the error.
         */

        return MatrixUtility.add(networkOutput, MatrixUtility.multiply(expected, -1));
    }

    private int getMaxIndex(double[] input){
        double max=0;
        int index=0;

        for(int i=0; i < input.length; i++){
            if(input[i] >= max){
                index = i;
                max = input[i];
            }
        }

        return index;
    }

    public int guess(Image image) throws Exception {
        List<double[][]> inList = new ArrayList<>();
        inList.add(MatrixUtility.multiply(image.data(), (1.0/scaleFactor)));

        double[] out = _layers.get(0).getOutput(inList);
        int guess = getMaxIndex(out);

        return guess;
    }

    public float test (List<Image> images) throws Exception {
        int correct = 0;

        for(Image img:images){
            int guess = guess(img);

            if(guess == img.label())
                correct++;
        }

        return (float)correct/images.size();
    }

    public void train(List<Image> images) throws Exception {
        for(Image img:images){
            List<double[][]> inList = new ArrayList<>();
            inList.add(MatrixUtility.multiply(img.data(), (1.0/scaleFactor)));

            double[] out = _layers.get(0).getOutput(inList);
            double[] dLdO = getErrors(out, img.label());

            _layers.get(_layers.size()-1).backPropagation(dLdO);
        }
    }
}
