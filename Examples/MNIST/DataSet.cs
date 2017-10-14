using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;
using ConvNetSharp.Volume.Double;
using ConvNetSharp.SNet;

namespace ATTFace
{
    internal class DataSet
    {
        private readonly List<ATTEntry> _trainImages;
        private readonly Random _random = new Random(RandomUtilities.Seed);
        private int _start;
        private int _epochCompleted;

        public DataSet(List<ATTEntry> trainImages)
        {
            this._trainImages = trainImages;
        }

        public Tuple<Volume, Volume, int[]> NextBatch(int batchSize)
        {
            const int w = 92;
            const int h = 112;

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
                    //50% of the time find a matching entry.
                    var findMatch = false;
                    
                    //If i is not odd, the match will not line up to be tried against each other.
                    if (i % 2 == 1)
                        findMatch = this._random.NextDouble() > 0.5;

                    if (findMatch && i > 0)
                    {
                        ATTEntry current = this._trainImages[i];
                        int randomIndex;
                        
                        do
                        {
                            //find a match.
                            randomIndex = this._random.Next(this._trainImages.Count);
                        }
                        while (current.Label != this._trainImages[randomIndex].Label || randomIndex == i);
                        
                        //swap match into next position.
                        ATTEntry temp = this._trainImages[randomIndex];
                        this._trainImages[randomIndex] = this._trainImages[--i];
                        this._trainImages[i] = temp;
                    }
                    else
                    {
                        var j = this._random.Next(i);
                        var temp = this._trainImages[j];
                        this._trainImages[j] = this._trainImages[i];
                        this._trainImages[i] = temp;
                    }
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


                if (this._start >= this._trainImages.Count)
                {
                    this._start = 0;
                    this._epochCompleted++;
                    Console.WriteLine($"Epoch #{this._epochCompleted}");
                }





                //25% of the time find a matching entry.
                //var findMatch = this._random.NextDouble() < 0.25;

                ATTEntry entry = this._trainImages[this._start++];
                ATTEntry entry2 = this._trainImages[this._start++];
                //if (findMatch)
                //{
                //    int randomIndex;
                //    do
                //    {
                //        //find a match.
                //        randomIndex = this._random.Next(this._trainImages.Count);
                //    }
                //    while (entry.Label != this._trainImages[randomIndex].Label);

                //    //swap match into position.
                //    ATTEntry temp = this._trainImages[randomIndex];
                //    this._trainImages[randomIndex] = this._trainImages[this._start];
                //    this._trainImages[this._start] = temp;
                //    entry2 = this._trainImages[this._start++];
                //}
                //else
                //{
                //    entry2 = this._trainImages[this._start++]; // Array out of index HERE
                //}

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
                var compare = 0.0;
                if (labels[i * 2] == labels[i * 2 + 1])
                    compare = 1.0;
                expected[i] = compare;

                //expected output is dim: [1, 1, 1, batchSize]. Either 1.0(no match), or 0.5 (match)
                //Do a label comparison of chosen input to determine expected output.

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