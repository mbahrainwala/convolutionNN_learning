package layers;

import java.util.List;
import java.util.Random;

public class FullyConnectedLayer extends Layer{
    private final double[][] _weights;
    private final int _inputLength;
    private final int _outputLength;
    private final double[] lastZ; //should be the size of _outputLength
    private double[] lastInput; //should be the size of _inputLength

    private final long SEED;
    private final double LEARNING_RATE;

    private static final double LEEK = 0.01;

    public FullyConnectedLayer(int inLenght, int outLength, long seed, double learningRate){
        this._inputLength = inLenght;
        this._outputLength = outLength;

        lastZ = new double[_outputLength];

        this._weights = new double[inLenght][outLength];
        this.SEED = seed;
        this.LEARNING_RATE = learningRate;

        setRandomWeights();
    }

    private double[] fullyConnectedForwardPass(double[] input){
        lastInput = input;

        for(int j=0; j< _outputLength; j++){
            for(int i=0; i< _inputLength; i++) {
                lastZ[j] += input[j] * _weights[i][j];
            }
        }

        double[] out = new double[_outputLength];

        for(int j=0; j< _outputLength; j++){
            out[j] = relu(lastZ[j]);
        }

        return out;
    }



    @Override
    public double[] getOutput(List<double[][]> input) throws Exception {
        return getOutput(fullyConnectedForwardPass(matrixToVector(input)));
    }

    @Override
    public double[] getOutput(double[] input) throws Exception {
        double[] forwardPass = fullyConnectedForwardPass(input);

        return get_nextLayer() !=null?get_nextLayer().getOutput(forwardPass):forwardPass;
    }

    @Override
    public void backPropagation(List<double[][]> dLdO) {
        backPropagation(matrixToVector(dLdO));
    }

    /**
     *  <b>NOTE:</b>
     *  What we want to do is to work out how much which of our weights contributed towards the error
     *  This is to train our NN
     *  <br/>
     *  1. Take each weight and add the losses to that wright.<br/>
     *  2. dL_dO is the difference in loss/difference in output. This is passed from the previous layer.<br/>
     *  3. dO_dZ is how much output depends on the z. This is 0 for z<0 and 1 for z>0 for a relU derivative.<br/>
     *  4. dZ_dW is how much z depends on W. This is simply the input of the layer<br/>
     *  5. z = Sum(input * weights) [z=sum(input*weight)].<br/>
     *  6. dL_dw = dL_dO*dO_dZ*dZ_dW. This is the cost associated with the incorrect answer<br/>
     *  7. LEARNING_RATE is introduced to make sure that the NN does not get biased for 1 input but gradually learns for all different inputs.<br/>
     *      This would cause an issue where it will only recognise a particular image set only as the desired outputs.<br/>
     */
    @Override
    public void backPropagation(double[] dL_dO) {
        double[] dl_dx = new double[_inputLength];
        double dO_dZ;
        double dZ_dW;
        double dL_dW;
        double dZ_dX;

        for(int k=0; k<_inputLength;  k++) {

            double dl_dx_sum = 0;

            for (int j = 0; j < _outputLength; j++) {
                dO_dZ = derivative_relu(lastZ[j]);
                dZ_dW = lastInput[j];

                dL_dW = dL_dO[j] * dO_dZ * dZ_dW; //this is the cost.

                dZ_dX = _weights[k][j];

                _weights[k][j] -= dL_dW * LEARNING_RATE;
                //subtract the cost from the weights
                // (multiply with a LEARNING_RATE to control how much you want the cost to quickly influence the NN)

                //we need to now propagate this to the previous layer too.
                //here the error is dz_dx and that is the previous wight (before correction)
                if(get_previousLayer() !=null) //only do this calculation if there is a previous layer
                    dl_dx_sum += dL_dO[j] * dO_dZ * dZ_dX;
            }

            dl_dx[k] = dl_dx_sum;
        }

        if(get_previousLayer() !=null)//back propagation if there is a previous layer
            get_previousLayer().backPropagation(dl_dx);
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
        return _outputLength;
    }

    public void setRandomWeights(){
        Random rand = new Random(this.SEED);

        for(int i=0; i<_inputLength; i++){
            for(int j=0; j<_outputLength; j++){
                this._weights[i][j] = rand.nextGaussian();
            }
        }
    }

    private double relu(double input){
        if(input <=0)
            return 0;
        else return input;
    }

    private double derivative_relu(double input){
        if(input <=0)
            return LEEK;
        else return 1;
    }
}
