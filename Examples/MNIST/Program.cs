using System;
using System.Linq;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers.Double;
using ConvNetSharp.Core.Training;
using ConvNetSharp.Volume.Double;
using ConvNetSharp.SNet;
using ConvNetSharp.SNet.Layers;
using System.Drawing;
using System.Collections.Generic;

namespace MNIST
{
    internal class Program
    {


        private readonly CircularBuffer<double> _testAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> _trainAccWindow = new CircularBuffer<double>(100);
        private readonly CircularBuffer<double> _lossWindow = new CircularBuffer<double>(100);
        private SNet<double> _snet;
        private int _stepCount;
        private SgdTrainer<double> _trainer;

        private DataSets datasets;

        private static void Main()
        {
            var program = new Program();
            program.MnistDemo();

            /**
             * TODO:
             * modify _net to _snet
             * Modify DataSets.cs to give correct input for snets.
             * update this.train() for double input
             * update accuracy measurement in this.test()
             * 
             */

        }

        private void MnistDemo()
        {
            this.datasets = new DataSets();
            if (!datasets.Load(100))
            {
                return;
            }

            // Create network
            this._snet = new SNet<double>();
            this._snet.AddLayer(new InputLayer(28, 28, 1));
            this._snet.AddLayer(new ConvLayer(5, 5, 8) { Stride = 1, Pad = 2 });
            this._snet.AddLayer(new ReluLayer());
            this._snet.AddLayer(new PoolLayer(2, 2) { Stride = 2 });
            this._snet.AddLayer(new ConvLayer(5, 5, 16) { Stride = 1, Pad = 2 });
            this._snet.AddLayer(new ReluLayer());
            this._snet.AddLayer(new PoolLayer(3, 3) { Stride = 3 });
            this._snet.AddLayer(new FullyConnLayer(10));
            //this._snet.AddLayer(new SoftmaxLayer(10));

            this._snet.AddDistanceLayer(new TwinJoinLayer()); //Closer to 0 the better.
            this._snet.AddDistanceLayer(new SigmoidLayer()); //Closer to 0.5 the better. 

            this._trainer = new SgdTrainer<double>(this._snet)
            {
                LearningRate = 0.01,
                BatchSize = 1,
                L2Decay = 0.005,
                //Momentum = 0.9
            };

            Console.WriteLine("Convolutional neural network learning...[Press any key to stop]");
            do
            {
                var trainSample = datasets.Train.NextBatch(this._trainer.BatchSize);
                Train(trainSample.Item1, trainSample.Item2, trainSample.Item3);

                var testSample = datasets.Test.NextBatch(this._trainer.BatchSize);
                Test(testSample.Item1, testSample.Item3, this._testAccWindow);

                this._lossWindow.Add(this._trainer.Loss);

                Console.WriteLine("Loss: {0} Train accuracy: {1}% Test accuracy: {2}%", this._trainer.Loss,
                    Math.Round(this._trainAccWindow.Items.Average() * 100.0, 2),
                    Math.Round(this._testAccWindow.Items.Average() * 100.0, 2));

                Console.WriteLine("Example seen: {0} Fwd: {1}ms Bckw: {2}ms", this._stepCount,
                    Math.Round(this._trainer.ForwardTimeMs, 2),
                    Math.Round(this._trainer.BackwardTimeMs, 2));
            } while (!Console.KeyAvailable);

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



                //why? why not? ;O
                //                          Are the labels the same?            Does Prediction match result?
            }

            bool displayImages = false;
            if (!displayImages) return;
            if (this._stepCount < 2000) return;

            SNet<double>.SplitVolumes(x, out ConvNetSharp.Volume.Volume<double> v1, out ConvNetSharp.Volume.Volume<double> v2);

            var numbers = VolumeToBitmap(v1 as Volume, 28, 28);
            var numbers2 = VolumeToBitmap(v2 as Volume, 28, 28);

            var popup = new DisplayImage();
            //int count = 0;
            //for (int i = this._stepCount; i < this._stepCount + predictions.Length * 2; i+= 2)
            //{
            //    Double res = (labels[count * 2] == labels[count * 2 + 1] ? 1.0 : 0.0) == (double)predictions[count] ? 1.0 : 0.0;
            //    popup.DisplayData(this.datasets.Train.getImage(i), labels[count], 
            //        this.datasets.Train.getImage(i + 1), labels[count + 1],
            //        res.ToString() );
            //    popup.ShowDialog();

            //    count++;

            //}

            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] != 1) continue;
                //double res = (labels[i * 2] == labels[i * 2 + 1] ? 1.0 : 0.0);
                popup.DisplayData(numbers[i], labels[i * 2], numbers2[i], labels[i * 2 + 1], predictions[i].ToString());
                popup.ShowDialog();
            }

        }

        private void Train(Volume x, Volume y, int[] labels)
        {
            this._trainer.Train(x, y);

            Test(x, labels, this._trainAccWindow, false);

            this._stepCount += labels.Length;
        }






        private List<Bitmap> VolumeToBitmap(Volume v, int width, int height)
        {
            List<Bitmap> bmps = new List<Bitmap>();

            ////Copy image into Volume.
            //var j = 0;
            //for (var y = 0; y < h; y++)
            //{
            //    for (var x = 0; x < w; x++)
            //    {
            //        dataVolume.Set(x, y, 0, i, entry.Image[j] / 255.0);
            //        dataVolume2.Set(x, y, 0, i, entry2.Image[j++] / 255.0);
            //    }
            //}

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