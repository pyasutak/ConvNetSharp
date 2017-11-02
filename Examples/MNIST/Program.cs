using System;
using System.Linq;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume.Double;
using ConvNetSharp.SNet;
using ConvNetSharp.SNet.Layers;
using System.Drawing;
using System.Collections.Generic;

namespace ATTFace
{
    internal class Program
    {
        private readonly CircularBuffer<double> _validAccWindow = new CircularBuffer<double>(20);
        private readonly CircularBuffer<double> _trainAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> _lossWindow = new CircularBuffer<double>(5);
        private SNet<double> _snet;
        private int _stepCount;
        private SgdTrainer<double> _trainer;

        private DataSets datasets;

        private static void Main()
        {
            var program = new Program();
            program.ATTDemo();
        }

        private void ATTDemo()
        {
            this.datasets = new DataSets();
            if (!datasets.Load(100))
            {
                return;
            }

            // Create network
            this._snet = new SNet<double>();
            this._snet.AddLayer(new InputLayer(92, 112, 1));                        //input shape:
            this._snet.AddLayer(new ConvLayer(5, 5, 16) { Stride = 1});             //92 x 112 x  1 x 20
            this._snet.AddLayer(new ReluLayer());                                   //92 x 112 x  8 x 20
            this._snet.AddLayer(new PoolLayer(2, 2) { Stride = 2 });                //92 x 112 x  8 x 20
            this._snet.AddLayer(new ConvLayer(5, 5, 32) { Stride = 1});             //46 x  56 x  8 x 20
            this._snet.AddLayer(new ReluLayer());                                   //46 x  56 x 16 x 20
            this._snet.AddLayer(new PoolLayer(2, 2) { Stride = 2 });                //46 x  56 x 16 x 20
            this._snet.AddLayer(new FullyConnLayer(1000));                           //23 x  28 x 16 x 20
            this._snet.AddLayer(new SigmoidLayer());

            this._snet.AddDistanceLayer(new TwinJoinLayer());
            this._snet.AddDistanceLayer(new SigmoidLayer());

            this._trainer = new SgdTrainer<double>(this._snet)
            {
                LearningRate = 0.05,
                BatchSize = 20,
                L2Decay = 0.01,
                //Momentum = 0.9
            };

            // Program Loop
            while (true)
            {
                // Do learning
                Console.WriteLine("Convolutional neural network learning...[Press any key to test net]");
                bool epoch;
                do
                {
                    var trainSample = datasets.Train.NextBatch(this._trainer.BatchSize, out epoch);
                    Train(trainSample.Item1, trainSample.Item2, trainSample.Item3);

                    //var testsample = datasets.Validation.NextBatch(this._trainer.BatchSize);
                    //Test(testsample.Item1, testsample.Item3, this._validAccWindow);

                    //this._lossWindow.Add(this._trainer.Loss);

                    Console.WriteLine("Loss: {0} Train accuracy: {1}%", this._trainer.Loss,
                        Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2));

                    Console.WriteLine("Pairs seen: {0} Fwd: {1}ms Bckw: {2}ms", this._stepCount / 2,
                        Math.Round(this._trainer.ForwardTimeMs, 2),
                        Math.Round(this._trainer.BackwardTimeMs, 2));
                    //} while (!Console.KeyAvailable);
                } while (!epoch);
                Console.WriteLine($"Epoch #{datasets.Train.Epoch}");

                // Do Testing
                // Run on accWindow / batchSize batches.
                Console.WriteLine("Testing current network.");

                //for (int i = 0; i < 5; i++)
                //{
                var testsample = datasets.Validation.NextBatch(this._trainer.BatchSize, out bool epochThrowaway);
                Test(testsample.Item1, testsample.Item3, this._validAccWindow);
                //}


                Console.WriteLine("Test: Loss: {0} Train accuracy: {1}%", this._trainer.Loss,
                    Math.Round(this._validAccWindow.Items.Average() * 100.0, 2));

                Console.WriteLine("Test: Fwd: {0}ms Bckw: {1}ms",
                    Math.Round(this._trainer.ForwardTimeMs, 2),
                    Math.Round(this._trainer.BackwardTimeMs, 2));

                //while (Console.KeyAvailable)
                //    Console.ReadKey(true);

                //Check Validation Loss for convergence.

                if (this._lossWindow.Count == this._lossWindow.Capacity)
                {
                    double threshold = 0.001;
                    double avg = this._lossWindow.Items.Average();
                    if (Math.Sqrt(Math.Pow((avg - this._trainer.Loss), 2.0)) < threshold) //Euclidean Distance
                    {


                        break;
                    }
                }
                this._lossWindow.Add(this._trainer.Loss);

                if (this.datasets.Train.Epoch >= 200)
                    break;
            }
            Console.WriteLine("Training is Done.");

            while (true)
            {
                // Do learning
                Console.WriteLine("Run on Validation Set...[Press any key to test net]");
                do
                {
                    var testsample = datasets.Validation.NextBatch(this._trainer.BatchSize, out bool epochThrowaway);
                    Test(testsample.Item1, testsample.Item3, this._validAccWindow);

                    //var testsample = datasets.Validation.NextBatch(this._trainer.BatchSize);
                    //Test(testsample.Item1, testsample.Item3, this._validAccWindow);

                    //this._lossWindow.Add(this._trainer.Loss);

                    Console.WriteLine("Loss: {0} Train accuracy: {1}%", this._trainer.Loss,
                        Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2));

                    Console.WriteLine("Fwd: {0}ms Bckw: {1}ms",
                        Math.Round(this._trainer.ForwardTimeMs, 2),
                        Math.Round(this._trainer.BackwardTimeMs, 2));
                } while (!Console.KeyAvailable);
            }
        }

        private void Train(Volume x, Volume y, int[] labels)
        {
            this._trainer.Train(x, y);

            Test(x, labels, this._trainAccWindow, false);

            this._stepCount += labels.Length;
        }



        private void Test(Volume x, int[] labels, CircularBuffer<double> accuracy, bool forward = true)
        {
            if (forward)
            {
                this._snet.Forward(x);
            }

            var predictions = this._snet.GetPrediction();

            for (var i = 0; i < predictions.Length; i++)
            {
                accuracy.Add(
                    (labels[i * 2] == labels[i * 2 + 1] ? 1.0 : 0.0)
                    == predictions[i]
                    ? 1.0 : 0.0);
            }
            

            //Show image toggle
            bool displayImages = false;
            

            if (!displayImages) return;
            //if (this._stepCount < 2000) return;

            SNet<double>.SplitVolumes(x, out ConvNetSharp.Volume.Volume<double> v1, out ConvNetSharp.Volume.Volume<double> v2);

            var numbers = VolumeToBitmap(v1 as Volume, 92, 112);
            var numbers2 = VolumeToBitmap(v2 as Volume, 92, 112);

            var popup = new DisplayImage();

            for (int i = 0; i < predictions.Length; i++)
            {
                //if (predictions[i] != 1) continue;
                //double res = (labels[i * 2] == labels[i * 2 + 1] ? 1.0 : 0.0);
                popup.DisplayData(numbers[i], labels[i * 2], numbers2[i], labels[i * 2 + 1], predictions[i].ToString());
                popup.ShowDialog();
            }

        }

        // Helper Function.
        private List<Bitmap> VolumeToBitmap(Volume v, int width, int height)
        {
            List<Bitmap> bmps = new List<Bitmap>();
            
            int batchSize = v.Shape.Dimensions[3];
            for (int n = 0; n < batchSize; n++)
            {
                Bitmap bmp = new Bitmap(width, height);
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int value = (int)(v.Get(x, y, 0, n) * 255);
                        bmp.SetPixel(x, y, Color.FromArgb(value, value, value));
                    }
                }
                bmps.Add(bmp);
            }

            return bmps;
        }



    }
}