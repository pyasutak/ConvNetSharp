using System;
using ConvNetSharp.SNet;
using ConvNetSharp.SNet.Layers;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training.Double;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;

namespace MinimalSnet
{
    class Program
    {
        private static void Main()
        {
            // species a 2-layer neural network with one hidden layer of 20 neurons
            var snet = new SNet<double>();

            // input layer declares size of input. here: 2-D data
            // ConvNetJS works on 3-Dimensional volumes (width, height, depth), but if you're not dealing with images
            // then the first two dimensions (width, height) will always be kept at size 1
            snet.AddLayer(new InputLayer(1, 1, 2));

            // declare 20 neurons
            snet.AddLayer(new FullyConnLayer(20));

            // declare a ReLU (rectified linear unit non-linearity)
            snet.AddLayer(new ReluLayer());

            // declare a fully connected layer that will be used by the softmax layer
            //snet.AddLayer(new FullyConnLayer(10));

            // declare the linear classifier on top of the previous hidden layer
            //snet.AddLayer(new SoftmaxLayer(10));

            // declare the join layer that joins the siamese-twin networks
            snet.AddDistanceLayer(new TwinJoinLayer());

            // declare a sigmoid layer that provides final output
            snet.AddDistanceLayer(new SigmoidLayer());

            // forward a random data point through the network
            var x = new Volume(new[] { 0.3, -0.5 }, new Shape(2));
            var x2 = new Volume(new[] { 0.3, -0.6 }, new Shape(2));

            var prob = snet.Forward(x, x2);

            // prob is a Volume. Volumes have a property Weights that stores the raw data, and WeightGradients that stores gradients
            Console.WriteLine("distance from ( 0.3, -0.5 ) to ( 0.3, -0.6 ) is: " + prob.Get(0)); // prints e.g. 0.50101

            //snet.ValidateTwin();

            var trainer = new SgdTrainer(snet) { LearningRate = 0.01, L2Decay = 0.001 };
            trainer.Train(SNet<double>.JoinVolumes(x, x2), new Volume(new[] { 0.1 }, new Shape(1, 1, 1, 1))); // train the network, specifying that x is class zero


            Console.WriteLine("Loss: {0} ", trainer.Loss);


            //snet.ValidateTwin();

            var prob2 = snet.Forward(x, x2);
            Console.WriteLine("distance from ( 0.3, -0.5 ) to ( 0.3, -0.6 ) is: " + prob2.Get(0));
            // now prints 0.50374, slightly higher than previous 0.50101: the networks
            // weights have been adjusted by the Trainer to give a higher probability to
            // the class we trained the network with (zero)


            //snet.ValidateTwin();

            Console.ReadLine();
        }
    }
}
