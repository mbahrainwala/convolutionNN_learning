package com.behrainwala;

import com.behrainwala.Network.NetworkBuilder;
import com.behrainwala.Network.NeuralNetwork;
import com.behrainwala.data.DataReader;
import com.behrainwala.data.Image;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        System.out.println("\n\n***Starting CNN***\n...loading data...\n");
        List<Image> imagesTest = DataReader.readData("data/mnist_test.csv");
        System.out.println("***Test Data Loaded***");
        System.out.println("Images in test data: "+imagesTest.size());
        List<Image> imagesTrain = DataReader.readData("data/mnist_train.csv");
        System.out.println("***Train Data Loaded***");
        System.out.println("Images in train data: "+imagesTrain.size());

        NetworkBuilder builder = new NetworkBuilder(28, 28, 256*100);
        long SEED = 123;
        builder.addConvolutionLayer(8, 5, 1, 0.1, SEED);
        builder.addMaxPoolLayer(3, 2);
        builder.addFullyConnectedLayer(10, 0.1, SEED);

        NeuralNetwork net = builder.build();

        try {
            float rate;
            rate = net.test(imagesTest);
            System.out.println("Pre training success rate: "+rate);

            int epochs = 3;
            for (int i = 0; i < epochs; i++) {
                Collections.shuffle(imagesTrain);
                net.train(imagesTrain);
                rate = net.test(imagesTest);
                System.out.println("Success rate after round " + i+1 + ": " + rate);
            }
        }catch (Exception e) {
            System.out.println(e.getMessage()+":"+ Arrays.stream(e.getStackTrace()).toList());
        }
    }
}