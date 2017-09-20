using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using ConvNetSharp.SNet;

namespace MNIST
{
    internal class DataSet
    {
        private readonly List<MnistEntry> _trainImages;
        private readonly Random _random = new Random(RandomUtilities.Seed);
        private int _start;
        private int _epochCompleted;

        public DataSet(List<MnistEntry> trainImages)
        {
            this._trainImages = trainImages;
        }

        public Tuple<Volume, Volume, int[]> NextBatch(int batchSize)
        {
            const int w = 28;
            const int h = 28;

            var dataShape = new Shape(w, h, 1, batchSize);
            var expectedShape = new Shape(1, 1, 1, batchSize);
            var data = new double[dataShape.TotalLength];
            var expected = new double[expectedShape.TotalLength];
            var labels = new int[batchSize * 2];

            // Shuffle for the first epoch
            if (this._start == 0 && this._epochCompleted == 0)
            {
                for (var i = this._trainImages.Count - 1; i >= 0; i--)
                {
                    var j = this._random.Next(i);
                    var temp = this._trainImages[j];
                    this._trainImages[j] = this._trainImages[i];
                    this._trainImages[i] = temp;
                }
            }

            var dataVolume = new Volume(data, dataShape);
            var dataVolume2 = new Volume(data, dataShape);

            for (var i = 0; i < batchSize; i++)
            {

                //How do I need to change this????
                /**
                 * Don't need batchsize. I am training to compare two single images.
                 *  Mainly I am unsure how my SNet will operate with batches.
                 *  If it works sufficiently with batches... can include.
                 *      (check snet.JoinVolume(), snet.SplitVolume(),
                 *      SVolume methods? I think that's the main issue?)
                 * The Input needs to be redone.
                 *  Either edit this method to output a second set for input, or combine for later seperation.
                 *  Which two inputs do I choose? Just two concecutive images? or shuffle? each epoch???
                 * The output needs to be redone.
                 *  Output is no longer by classes, but by comparison, so one output.
                 * Labels still outputed?
                 *  Could just output consecutive labels, so labels[2 * batchsize]
                 *  used to compare for accuracy... right?
                 *  
                 * Just get it done... can be improved later, eh?
                 * 
                 * 
                 * 
                 */

                //50% of the time find a matching entry.
                bool findMatch = this._random.NextDouble() < 0.5;

                var entry = this._trainImages[this._start++];
                MnistEntry entry2;
                if (findMatch)
                {
                    do
                    {
                        int randomIndex = this._random.Next(this._trainImages.Count);
                        entry2 = this._trainImages[randomIndex];
                    }
                    while (entry.Label != entry2.Label);
                }
                else
                {
                    entry2 = this._trainImages[this._start++];
                }

                //Store image label.
                labels[i * 2] = entry.Label;
                labels[i * 2 + 1] = entry2.Label;

                //Copy image into Volume.
                var j = 0;
                for (var y = 0; y < h; y++)
                {
                    for (var x = 0; x < w; x++)
                    {
                        dataVolume.Set(x, y, 0, i, entry.Image[j] / 255.0);
                        dataVolume2.Set(x, y, 0, i, entry2.Image[j++] / 255.0);
                    }
                }

                //Store expected output.
                double compare = 1.0; //Not the same label.
                if (labels[i * 2] == labels[i * 2 + 1])
                    compare = 0.5; //The same label
                expected[i] = compare;

                //expected output is dim: [1, 1, 1, batchSize]. 
                //Do a label comparison of chosen input to determine expected output.

                if (this._start >= this._trainImages.Count - batchSize)
                {
                    this._start = 0;
                    this._epochCompleted++;
                    Console.WriteLine($"Epoch #{this._epochCompleted}");
                }
            }


            var expectedVolume = new Volume(expected, expectedShape);
            var joinedData = SNet<double>.JoinVolumes(dataVolume, dataVolume2) as ConvNetSharp.Volume.Double.Volume;

            return new Tuple<Volume, Volume, int[]>(joinedData, expectedVolume, labels);
        }

        public byte[] getImage(int index)
        {
            return this._trainImages[index].Image;
        }

    }
}