/*
Am folosit frameworkul Jama (java matrix applications ) pentru a ne ajuta si a ne simplifica munca
I used the Jama framework (java matrix applications) for helping the calculations, given that alot of matrix math operations are already implemented

 */
package neuralnetwork2.pkg0;

import Jama.*;
import java.io.FileNotFoundException;
import java.io.UnsupportedEncodingException;


/**
 *
 * @author macair
 */
public class NeuralNetwork20 extends JFrame {

    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double dsigmoid(double x) {
        return x * (1 - x);
    }

    public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {

        Scanner scin = new Scanner(new FileReader("/Users/macair/Desktop/NeuralNetwork/Input/Input.txt"));
        scin.useLocale(Locale.US);

        Random random = new Random();
        NeuralNetwork20 math = new NeuralNetwork20();

        //declarations (toate)
        float cost = 0;
        float rightAnswer = 0;
        float[] finalAnswer = new float[2]; # binary classifications of ekgs
        int guess = 0;
        float accuracy = 0;

        Matrix input = new Matrix(200, 1);
        Matrix layer = new Matrix(100, 1);
        Matrix output = new Matrix(10, 1);

        Matrix wih = new Matrix(100, 200); /// wil = weights between input layer and hidden layer
        Matrix who = new Matrix(10, 100); // who = weights between h(hidden) layer and o(output) layer;

        Matrix eo = new Matrix(10, 1);//gradient of input layer
        Matrix eh = new Matrix(100, 1);//gradient of hidden layer

        Matrix errors = new Matrix(10, 1);
        Matrix h_errors = new Matrix(100, 1);

        Matrix dwih = new Matrix(200, 100);
        Matrix dwho = new Matrix(100, 10);

        for (int i = 1; i < 2; i++) {
            finalAnswer[i] = 0;
        }
        finalAnswer[0] = 1;

        //initializing the weights with randon samples from a uniform distribution ranging -d/100 si d/100 (d = number of neurons in hidden);
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 200; j++) {
                wih.set(i, j, (2 * random.nextFloat() - 1) / 100);
            }
        }

        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 100; j++) {
                who.set(i, j, (2 * random.nextFloat() - 1) / 100);
            }
        }

        //forward propagation
        do {

            //reading 200 values for input
            for (int i = 0; i < 200; i++) {

                input.set(i, 0, scin.nextFloat());
            }

            //label is 201th value
            rightAnswer = scin.nextFloat();

            // forward pass to hidden layer
            layer = wih.times(input);

            // applying sigmoid function
            for (int i = 0; i < 100; i++) {
                layer.set(i, 0, math.sigmoid(layer.get(i, 0)));
            }
            // output layer forward pass
            output = who.times(layer);

            // applying sigmoid function
            for (int i = 0; i < 10; i++) {
                output.set(i, 0, math.sigmoid(output.get(i, 0)));
            }

            double max = 0;

            // getting the maximum activation
            for (int i = 0; i < 2; i++) {
                if (output.get(i, 0) > (double) max) {
                    guess = i;
                    max = output.get(i, 0);
                }
            }

            //calculating erros and cost
            for (int i = 0; i < 10; i++) {
                cost += Math.pow((output.get(i, 0) - finalAnswer[i]), 2);
                errors.set(i, 0, (output.get(i, 0) - finalAnswer[i]));
            }

            h_errors = who.transpose().times(errors);

            // applying derivative of sigmoid
            for (int i = 0; i < 2; i++) {
                output.set(i, 0, math.dsigmoid(output.get(i, 0)));
            }

            //calculating gradient times learning rate
            dwho = errors.arrayTimes(output).times(layer.transpose());
            dwho.timesEquals(0.01);


            // applying derivative of sigmoid
            for (int i = 0; i < 100; i++) {
                layer.set(i, 0, math.dsigmoid(layer.get(i, 0)));
            }

            //calculating gradient times learning rate
            dwih = h_errors.arrayTimes(layer).times(input.transpose());
            dwih.timesEquals(0.01);//learning

            // substracting gradients
            wih.minusEquals(dwih);
            who.minusEquals(dwho);

            guess = 0;
            max = 0;
            cost = 0;
        } while (scin.hasNext());

    }

}
