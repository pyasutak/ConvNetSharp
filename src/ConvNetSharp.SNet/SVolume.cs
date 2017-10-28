using System;
using System.Collections.Generic;
using System.Text;
using ConvNetSharp.Volume;

namespace ConvNetSharp.SNet
{
    //Doesn't REALLY need to extend Volume, does it?
    public class SVolume : ConvNetSharp.Volume.Double.Volume
    {

        public VolumeStorage<double> StorageTwin { get; }

        public Shape ShapeTwin => this.StorageTwin.Shape;


        public SVolume(double[] array, Shape shape, double[] arrayTwo, Shape shapeTwo) :
            this(new NcwhVolumeStorage<double>(array, shape), new NcwhVolumeStorage<double>(arrayTwo, shapeTwo))
        {
        }

        public SVolume(VolumeStorage<double> storage, VolumeStorage<double> storageTwo) : base(storage)
        {
            this.StorageTwin = storageTwo;
        }

        //public static SVolume FromVolume(ConvNetSharp.Volume.Volume<double> source)
        //{
        //    var data = new double[source.Shape.TotalLength];
        //    Array.Copy(source.Storage.ToArray(), data, data.Length);
        //    return new SVolume(data, source.Shape);
        //}

        //public ConvNetSharp.Volume.Volume<double> ToVolume()
        //{
        //    var data = new double[this.Shape.TotalLength];
        //    Array.Copy(ToArray(), data, data.Length);

        //    return BuilderInstance<double>.Volume.SameAs(data, this.Shape);
        //}

        public void DoNetworkJoin(Volume<double> alpha, Volume<double> result)
        {
            // Need to consider input/output dimensions for twinjoinlayer. I think it's handled.
            // Currently it will compare and sum... maybe split summation into a new layer?
            // Join = Sum( alpha * abs( h1 - h2))
            int batchSize = this.Shape.GetDimension(3);

            int inputWidth = this.Shape.GetDimension(0);
            int inputHeight = this.Shape.GetDimension(1);
            int inputDepth = this.Shape.GetDimension(2);

            //Could link to output dimensions... but currently not needed.
            int sumWidth = 0;
            int sumHeight = 0;
            int sumDepth = 0;

            result.Clear();

            /**
             * 
             * FOR EVERY: input (width, height, depth)
             * sum += alpha[i] * | storage[i] - storageTwin[i] |
             * 
             * 
             * 
             */

            for (int n = 0; n < batchSize; n++)
            {
                for (int depth = 0; depth < inputDepth; depth++)
                {
                    for (int height = 0; height < inputHeight; height++)
                    {
                        for (int width = 0; width < inputWidth; width++)
                        {
                            double join = alpha.Get(width, height, depth, 0)
                                * Math.Abs((this.Storage.Get(width, height, depth, n) - this.StorageTwin.Get(width, height, depth, n)));
                            result.Set(sumWidth, sumHeight, sumDepth, n, (result.Get(0, 0, 0, n) + join));
                        }
                    }
                }
            }


        }

        public void DoNetworkJoinGradients(Volume<double> alpha, Volume<double> outputGradients,
            Volume<double> inputGradients, Volume<double> inputTwinGradients, Volume<double> alphaGradients)
        {
            // How do I do the alphaGradients???
            // Join = Sum( alpha * abs( h1 - h2))
            // JoinGradients = (alpha * abs(h1-h2)) / (h1-h2)) * outputGradients???
            // dh1 = -dh2???

            /**
             * What do I need to do here?
             * Compute:
             * InputGradients, InputTwinGradients, alphaGradients
             * From:
             * Alpha, Storage, StorageTwin, outputGradients.
             *                                                                                   Only one output, so only one gradient.
             * For (x[i] = storage[i]; y[i] = storageTwin[i]; a[i] = alpha[i]; OutputGradients = outputGradients[0])
             * InputGradients[i] = a[i] * ( (x[i] - y[i]) / Math.Abs(x[i] - y[i]) ) * OutputGradients
             * InputTwinGradients[i] = a[i] * ( (-1) * (x[i] - y[i]) / Math.Abs(x[i] - y[i]) ) * OutputGradients
             * Alpha[i] = Math.Abs(x[i] - y[i]) * OutputGradients
             * 
             * Loop for every i (every input)
             * 
             */
            int batchSize = this.Shape.GetDimension(3);

            int inputWidth = this.Shape.GetDimension(0); //always 1
            int inputHeight = this.Shape.GetDimension(1); //always 1
            int inputDepth = this.Shape.GetDimension(2); //input count = this.InputWidth * this.InputHeight * this.InputDepth;

            //Could link to output dimensions... but currently not needed.

            inputGradients.Clear();
            inputTwinGradients.Clear();
            alphaGradients.Clear();



            for (int n = 0; n < batchSize; n++)
            {
                double ChainGradient = outputGradients.Get(0, 0, 0, n);

                for (int width = 0; width < inputWidth; width++)
                {
                    for (int height = 0; height < inputHeight; height++)
                    {
                        for (int depth = 0; depth < inputDepth; depth++)
                        {
                            double a = alpha.Get(width, height, depth, 0);
                            double x = Storage.Get(width, height, depth, n);
                            double y = StorageTwin.Get(width, height, depth, n);

                            double aval, xval, yval;

                            if (x - y == 0) //Prevents the same image from breaking the snet.
                            {
                                xval = 0;
                                yval = 0;
                                aval = 0;
                            }
                            else
                            {
                                xval = (a * ((x - y) / Math.Abs(x - y))) * ChainGradient;
                                yval = (a * (-(x - y) / Math.Abs(x - y))) * ChainGradient;
                                aval = (Math.Abs(x - y)) * ChainGradient;
                            }

                            inputGradients.Set(width, height, depth, n,
                                inputGradients.Get(width, height, depth, n) + xval);
                            inputTwinGradients.Set(width, height, depth, n,
                                inputTwinGradients.Get(width, height, depth, n) + yval);
                            alphaGradients.Set(width, height, depth, 0,
                                alphaGradients.Get(width, height, depth, 0) + aval);
                        }
                    }
                }
            }






        }


    }
}
