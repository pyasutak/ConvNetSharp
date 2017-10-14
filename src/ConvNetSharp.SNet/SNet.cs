using System;
using System.Collections.Generic;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.SNet.Layers;

namespace ConvNetSharp.SNet
{
    public class SNet<T> : INet<T> where T : struct, IEquatable<T>, IFormattable
    {

        /**
         * Things to do:
         * 
         * Go over IsTwinValid(); done?
         * Go over Balance Gradients(); replaced for BalanceParameters()
         * Backward() is not implementing loss (through ILastLayer)
         * GetCostLoss(), getPrediction(), dump() not implemented.
         * FromData() and constructor through dic not customized.
         * 
         * 
         * 
         * 
         */

        public List<LayerBase<T>> Layers { get; } = new List<LayerBase<T>>();
        public List<LayerBase<T>> LayersTwin { get; } = new List<LayerBase<T>>();

        public List<LayerBase<T>> DistanceLayers { get; } = new List<LayerBase<T>>();


        public Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            //Assume the two inputs are given as a single Volume.
            //First, split the input, then call forward(input,input2);
            SplitVolumes(input, out Volume<T> split1, out Volume<T> split2);

            return this.Forward(split1, split2, isTraining);

            //return Forward(input, input, isTraining);
        }

        public Volume<T> Forward(Volume<T>[] input, bool isTraining = false)
        {
            return this.Forward(input[0], input[1], isTraining);
        }

        public Volume<T> Forward(Volume<T> inputA, Volume<T> inputB, bool isTraining = false)
        {
            //If isTraining is true, check to make sure parameters are updated.
            //Doesn't quite work... Needs to be called one more time after training. Possibly better solution.
            //if (isTraining)
            //{
            try
            {
                //Obsolete now?
                ValidateTwin();
            }
            catch
            {
                BalanceParameters();
                //Test
                //ValidateTwin();
            }
            //}

            var activationA = this.Layers[0].DoForward(inputA, isTraining);
            var activationB = this.LayersTwin[0].DoForward(inputB, isTraining);


            for (var i = 1; i < this.Layers.Count; i++)
            {
                var layer = this.Layers[i];
                var twinLayer = this.LayersTwin[i];
                activationA = layer.DoForward(activationA, isTraining);
                activationB = twinLayer.DoForward(activationB, isTraining);
            }

            //Use activationA and activataionB for the distance layer.

            var joinLayer = this.DistanceLayers[0] as IJoinLayer;
            if (joinLayer == null) throw new Exception("The first layer of the distance layer must join the networks.");
            var outA = activationA as Volume<double>;
            var outB = activationB as Volume<double>;

            var distance = joinLayer.DoForward(isTraining, outA, outB) as Volume<T>; //Urgh, the generics....

            for (var i = 1; i < this.DistanceLayers.Count; i++)
            {
                var distanceLayer = this.DistanceLayers[i];
                distance = distanceLayer.DoForward(distance, isTraining);
            }

            return distance;


        }

        public T Backward(Volume<T> expectedOutput)
        {
            var n = this.Layers.Count;
            var dn = this.DistanceLayers.Count;
            var lastLayer = this.Layers[n - 1];
            var lastLayerTwin = this.LayersTwin[n - 1];
            var lastDistanceLayer = this.DistanceLayers[dn - 1];
            if (lastLayer != null && lastLayerTwin != null) //This doesn't do anything. 
            {
                //TODO: I cut out implementation for loss. Requires ILastLayer interface. done, see below
                //Or it would... if the ILastLayer interface did anything. Loss is now implemented in this method.


                //whattehfux does this code do?:

                //for (int i = 0; i < expectedOutput.Shape.Dimensions[3]; i++)
                //    if (Ops<T>.GreaterThan(expectedOutput.Get(0, 0, 0, i), Ops<T>.Zero))
                //        break;

                //Preprocess expectedOutput wrt distance.
                //Volume<T> snetExpected = this.DistanceBackward(expectedOutput); //Review implementation!!
                //Above should return two Volumes, for either network.

                lastDistanceLayer.Backward(expectedOutput);
                for (int i = dn - 2; i >= 0; i--)
                {
                    this.DistanceLayers[i].Backward(this.DistanceLayers[i + 1].InputActivationGradients);
                }

                //Might want to add some checks here......
                //split SVolume into two Volumes for standard processing.
                var distanceLayerTwinGradients = (this.DistanceLayers[0] as TwinJoinLayer).InputTwinActivationGradients as Volume<T>;

                lastLayer.Backward(this.DistanceLayers[0].InputActivationGradients);
                lastLayerTwin.Backward(distanceLayerTwinGradients);
                this.BalanceGradients(n - 1);
                for (int i = n - 2; i >= 0; i--)
                {
                    // first layer assumed input //What does this meaaaaaan???
                    this.Layers[i].Backward(this.Layers[i + 1].InputActivationGradients);
                    this.LayersTwin[i].Backward(this.LayersTwin[i + 1].InputActivationGradients);

                    //Balance Gradients --- balance everything at the end????
                    this.BalanceGradients(i);
                }


                //Depreciate this.
                //Balance Gradients
                //for (int i = 0; i < n; i++)
                //    BalanceGradients(i);
                //BalanceParameters();



                //Calculate Loss.
                T loss = Ops<T>.Zero;
                var y = expectedOutput;

                for (var N = 0; N < y.Shape.GetDimension(3); N++)
                {
                    for (var d = 0; d < y.Shape.GetDimension(2); d++)
                    {
                        for (var h = 0; h < y.Shape.GetDimension(1); h++)
                        {
                            for (var w = 0; w < y.Shape.GetDimension(0); w++)
                            {
                                var expected = y.Get(w, h, d, N);
                                var actual = lastDistanceLayer.OutputActivation.Storage.Get(w, h, d, N);
                                if (Ops<T>.Zero.Equals(actual))
                                    actual = Ops<T>.Epsilon;
                                var current = Ops<T>.Multiply(expected, Ops<T>.Log(actual));

                                loss = Ops<T>.Add(loss, current);
                            }
                        }
                    }
                }
                loss = Ops<T>.Negate(loss);



                ////Contrastive Loss
                //var euclideanDistance = this.DistanceLayers[0].OutputActivation.Storage.Get(0);
                //T m = Ops<T>.Cast(1.0);
                //for (var N = 0; N < y.Shape.GetDimension(3); N++)
                //{
                //    for (var d = 0; d < y.Shape.GetDimension(2); d++)
                //    {
                //        for (var h = 0; h < y.Shape.GetDimension(1); h++) //always 1
                //        {
                //            for (var w = 0; w < y.Shape.GetDimension(0); w++) //always 1
                //            {
                //                var expected = y.Get(w, h, d, N);
                //                if (Ops<T>.GreaterThan(Ops<T>.One, expected))
                //                    expected = Ops<T>.Zero;
 
                //                var match = Ops<T>.Multiply(Ops<T>.Subtract(Ops<T>.One, expected), Ops<T>.Multiply(Ops<T>.Cast(0.5), Ops<T>.Pow(euclideanDistance, Ops<T>.Cast(2.0))));
                //                var nomatch = Ops<T>.Multiply(expected, Ops<T>.Multiply(Ops<T>.Cast(0.5), Ops<T>.Pow(Ops<T>.Subtract(Ops<T>.One, euclideanDistance), Ops<T>.Cast(2.0))));
                //                if (!Ops<T>.GreaterThan(nomatch, Ops<T>.Zero))
                //                    nomatch = Ops<T>.Epsilon;
 
                //                var current = Ops<T>.Add(match,nomatch);
 
                //                //var actual = joinLayer.OutputActivation.Storage.Get(w, h, d, N);
                //                //if (Ops<T>.Zero.Equals(actual))
                //                //    actual = Ops<T>.Epsilon;
                //                //var current = Ops<T>.Multiply(expected, Ops<T>.Log(actual));
 
                //                loss = Ops<T>.Add(loss, current);
                //            }
                //        }
                //    }
                //}
 




                return loss;
            }

            throw new Exception("The SNet is empty!");
        }

        //UPDATE
        public void ValidateTwin()
        {
            /**
             * Constant values for LayerBase:
             * InputActivation
             * OutputActivation
             * InputActivationGradients
             * OutputActivationGradients
             * Input/Output Height/Width/Depth
             * 
             * 
             * 
             * The thing to check is Parameters and Gradients, Foolish past self.
             * Do I really need to check all of the minor class variables for each layer?
             *  They are not changed once initialized... but it might be important to check each layer for consistency...???
             * 
             * 
             * 
             */






            Func<T, T, T> verify = (x, y) => Ops<T>.Equals(x, y) ? x : throw new Exception("The Siamese Net is invalid.");

            for (int i = 0; i < Layers.Count; i++)
            {
                var l = Layers[i];
                var lt = LayersTwin[i];



                //Really want to check these????
                //var inGradientsVerification = BuilderInstance<T>.Volume.SameAs(l.InputActivationGradients.Shape).Storage;
                //l.InputActivationGradients.Storage.Map(verify, lt.InputActivationGradients.Storage, inGradientsVerification);
                //var outGradientsVerification = BuilderInstance<T>.Volume.SameAs(l.OutputActivationGradients.Shape).Storage;
                //l.OutputActivationGradients.Storage.Map(verify, lt.OutputActivationGradients.Storage, outGradientsVerification);

                /*
                 * convlayer
                 * fullyconnlayer
                 *  :Filter, Bias
                 *  
                 * PoolLayer
                 *  :Width, Height, Stride, Pad
                 * 
                 * 
                 */

                var convl = l as ConvLayer<T>;
                if (convl != null)
                {
                    var convlt = lt as ConvLayer<T>;
                    if (convlt == null) throw new Exception("The Layers in the Siamese Net are not valid!");

                    var filtersVerification = BuilderInstance<T>.Volume.SameAs(convl.Filters.Shape).Storage;
                    convl.Filters.Storage.Map(verify, convlt.Filters.Storage, filtersVerification);
                    var biasVerification = BuilderInstance<T>.Volume.SameAs(convl.Bias.Shape).Storage;
                    convl.Bias.Storage.Map(verify, convlt.Bias.Storage, biasVerification);
                }

                var connl = l as FullyConnLayer<T>;
                if (connl != null)
                {
                    var connlt = lt as FullyConnLayer<T>;
                    if (connlt == null) throw new Exception("The Layers in the Siamese Net are not valid!");

                    var filtersVerification = BuilderInstance<T>.Volume.SameAs(connl.Filters.Shape).Storage;
                    connl.Filters.Storage.Map(verify, connlt.Filters.Storage, filtersVerification);
                    var biasVerification = BuilderInstance<T>.Volume.SameAs(connl.Bias.Shape).Storage;
                    connl.Bias.Storage.Map(verify, connlt.Bias.Storage, biasVerification);
                }

                var pl = l as PoolLayer<T>;
                if (pl != null)
                {
                    var plt = lt as PoolLayer<T>;
                    if (plt == null) throw new Exception("The Layers in the Siamese Net are not valid!");

                    if (pl.Width != plt.Width) throw new Exception("Poll Layer's Widths do not match.");
                    if (pl.Height != plt.Height) throw new Exception("Poll Layer's Heights do not match.");
                    if (pl.Stride != plt.Stride) throw new Exception("Poll Layer's Strides do not match.");
                    if (pl.Pad != plt.Pad) throw new Exception("Poll Layer's Pads do not match.");
                }

            }
        }


        private void BalanceGradients(int layerIndex)
        {
            


            var pandg = Layers[layerIndex].GetParametersAndGradients();
            var pandgT = LayersTwin[layerIndex].GetParametersAndGradients();

            for (int i = 0; i < pandg.Count; i++)
            {


                //}
                var gradient = pandg[i].Gradient;
                var gradientT = pandgT[i].Gradient;
                var temp = gradient.Clone();

                //Verify dimensions??? Why even bother at this point?

                //var width = gradient.Shape.GetDimension(0);
                //var height = gradient.Shape.GetDimension(1);
                //var depth = gradient.Shape.GetDimension(2);
                //var batch = gradient.Shape.GetDimension(3);


                //var avg = BuilderInstance<T>.Volume.SameAs(gradient.Shape);
                //gradient.Storage.Map((x, y) => Ops<T>.Divide(Ops<T>.Add(x, y), Ops<T>.Cast(2)), gradientT.Storage, avg.Storage);

                //var avg = gradient + gradientT * Ops<T>.Cast(0.5);

                gradient.DoAdd(gradientT, gradient);
                //gradient.DoMultiply(gradient, Ops<T>.Cast(0.5)); //Additive instead of average
                gradientT.DoAdd(temp, gradientT);
                //gradientT.DoMultiply(gradientT, Ops<T>.Cast(0.5));

                //this does nothing. lel. The only gradients that are trained upon are those passed in GetParametersAndGradients()
                //better now. wew

            }

            
            //RETHINK THIS PROCESS...
            /*
             * Rebalancing the gradients at every level makes the entire point of the siamese network irrelevant.
             * Rebalance gradients at the end of the backward.
             * What needs to be rebalanced? EVERYTHING???
             * Everything that needs to be verified.
             * I.E. everything that backwards() can change.
             * 
             * Could use:
             * GetParametersAndGradients()
             * to balance the layers.
             * Ultimately, the actual values in the layers are in the parameters and gradients.
             * 
             * However... The training algorithm updates the layers based upon the inputactivationgradients... so those should be used.
             * As long as the training algorithm sees that the twin layers have the same inputActivationGradients, they should be 
             *  updated to the same values.
             *  
             *  Basically, leave as is?
             */



        }



        private void BalanceParameters()
        {
            Func<T, T, T> avg = (x, y) => Ops<T>.Divide(Ops<T>.Add(x, y), Ops<T>.Cast(2.0));
            //Func<T, T> copy = x => x;



            //Func<T, T, T> verify = (x, y) => Ops<T>.Equals(x, y) ? x : throw new Exception("The Siamese Net is invalid.");



            for (int i = 0; i < this.Layers.Count; i++)
            {
                var layerParams = this.Layers[i].GetParametersAndGradients();
                var layerTwinParams = this.LayersTwin[i].GetParametersAndGradients();

                for (int j = 0; j < layerParams.Count; j++)
                {
                    //Average Volume.
                    Volume<T> average = BuilderInstance<T>.Volume.SameAs(layerParams[j].Volume.Shape);
                    layerParams[j].Volume.Storage.Map(avg, layerTwinParams[j].Volume.Storage, average.Storage);
                    layerParams[j].Volume.Storage.CopyFrom(average.Storage);
                    layerTwinParams[j].Volume.Storage.CopyFrom(average.Storage);
                    //Average Gradient.
                    average = BuilderInstance<T>.Volume.SameAs(layerParams[j].Gradient.Shape);
                    layerParams[j].Gradient.Storage.Map(avg, layerTwinParams[j].Volume.Storage, average.Storage);
                    layerParams[j].Gradient.Storage.CopyFrom(average.Storage);
                    layerTwinParams[j].Gradient.Storage.CopyFrom(average.Storage);
                }

            }

        }


        public void AddLayer(LayerBase<T> layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            LayerBase<T> lastLayer = null;

            if (this.Layers.Count > 0)
            {
                inputWidth = this.Layers[this.Layers.Count - 1].OutputWidth;
                inputHeight = this.Layers[this.Layers.Count - 1].OutputHeight;
                inputDepth = this.Layers[this.Layers.Count - 1].OutputDepth;
                lastLayer = this.Layers[this.Layers.Count - 1];
            }
            else if (!(layer is InputLayer<T>))
            {
                throw new ArgumentException("First layer should be an InputLayer");
            }

            var classificationLayer = layer as IClassificationLayer;
            if (classificationLayer != null)
            {
                var fullconLayer = lastLayer as FullyConnLayer<T>;
                if (fullconLayer == null)
                {
                    throw new ArgumentException(
                        $"Previously added layer should be a FullyConnLayer with {classificationLayer.ClassCount} Neurons");
                }

                if (fullconLayer.NeuronCount != classificationLayer.ClassCount)
                {
                    throw new ArgumentException(
                        $"Previous FullyConnLayer should have {classificationLayer.ClassCount} Neurons");
                }
            }

            var reluLayer = layer as ReluLayer<T>;
            if (reluLayer != null)
            {
                var dotProductLayer = lastLayer as IDotProductLayer<T>;
                if (dotProductLayer != null)
                {
                    // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.

                    dotProductLayer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?
                    //Update Twin.
                    var twinDotProductLayer = this.LayersTwin[this.Layers.Count - 1] as IDotProductLayer<T>;
                    twinDotProductLayer.BiasPref = dotProductLayer.BiasPref;
                    //Both ConvLayer and FullyConnLayer have filters that are updated by this. Copy over to twin.

                    //Apparntly the below code breaks the program... 
                    //I think having the filters be the same results in too many distances of x-y = 0... which is a 
                    //divide by 0 issue in my math in SVolume.

                    if (this.Layers[this.Layers.Count - 1] is ConvLayer<T> l && this.LayersTwin[this.Layers.Count - 1] is ConvLayer<T> lt)
                    {
                        lt.Filters.Storage.Map(((x, y) => y), l.Filters.Storage, lt.Filters.Storage);
                    }
                    if (this.Layers[this.Layers.Count - 1] is FullyConnLayer<T> lc && this.LayersTwin[this.Layers.Count - 1] is FullyConnLayer<T> lct)
                    {
                        lct.Filters.Storage.Map(((x, y) => y), lc.Filters.Storage, lct.Filters.Storage);
                    }
                }
            }

            //var layertwin = LayerBase<T>.FromData(layer.GetData()); //Not good enough!!!!

            if (this.Layers.Count > 0)
            {
                layer.Init(inputWidth, inputHeight, inputDepth);
            }

            //var layertwin = CloneLayer(layer);
            var layertwin = LayerBase<T>.FromData(layer.GetData());

            this.Layers.Add(layer);
            this.LayersTwin.Add(layertwin);
        }

        //Should only be used after finished adding normal net layers.
        public void AddDistanceLayer(LayerBase<T> layer)
        {
            int inputWidth = 0, inputHeight = 0, inputDepth = 0;
            LayerBase<T> lastLayer = null;

            if (this.DistanceLayers.Count > 0)
            {
                inputWidth = this.DistanceLayers[this.DistanceLayers.Count - 1].OutputWidth;
                inputHeight = this.DistanceLayers[this.DistanceLayers.Count - 1].OutputHeight;
                inputDepth = this.DistanceLayers[this.DistanceLayers.Count - 1].OutputDepth;
                lastLayer = this.DistanceLayers[this.DistanceLayers.Count - 1];
            }
            else if (!(layer is IJoinLayer))
            {
                throw new ArgumentException("First distance layer should be a Join Layer");
            }
            else if (this.DistanceLayers.Count == 0)
            {
                inputWidth = this.Layers[this.Layers.Count - 1].OutputWidth;
                inputHeight = this.Layers[this.Layers.Count - 1].OutputHeight;
                inputDepth = this.Layers[this.Layers.Count - 1].OutputDepth;
                lastLayer = this.Layers[this.Layers.Count - 1];
            }

            //This can change. The distance layer will be receiving input.
            layer.Init(inputWidth, inputHeight, inputDepth);

            this.DistanceLayers.Add(layer);
        }



        //Obsolete.



        ////To be removed:
        //private static LayerBase<T> CloneLayer(LayerBase<T> source)
        //{

        //    //var layers = dico["Layers"] as IEnumerable<IDictionary<string, object>>;
        //    //foreach (var layerData in layers)
        //    //{
        //    //    var layer = LayerBase<T>.FromData(layerData);
        //    //    net.Layers.Add(layer);
        //    //}

        //    //Copies all class properties that are init when created. Copies all parameters and gradients.


        //    var conv = source as ConvLayer<T>;
        //    if (conv != null)
        //    {
        //        conv.GetData();
        //        var clone = new ConvLayer<T>(conv.Width, conv.Height, conv.FilterCount);
        //        clone.BiasPref = conv.BiasPref;

        //        //copy parameters and gradients
        //        //var pg = source.GetParametersAndGradients();
        //        //var clonepg = clone.GetParametersAndGradients();

        //        //for (int i = 0; i < pg.Count; i++)
        //        //{
        //        //    clonepg[i].Volume = BuilderInstance<T>.Volume.SameAs(pg[i].Volume.Storage, pg[i].Volume.Shape);
        //        //    clonepg[i].Gradient = BuilderInstance<T>.Volume.SameAs(pg[i].Gradient.Storage, pg[i].Gradient.Shape);
        //        //    clonepg[i].L1DecayMul = pg[i].L1DecayMul;
        //        //    clonepg[i].L2DecayMul = pg[i].L2DecayMul;
        //        //}

        //        //Volume = this.Filters,
        //        //    Gradient = this.FiltersGradient,
        //        //    L2DecayMul = this.L2DecayMul,
        //        //    L1DecayMul = this.L1DecayMul
        //        //},
        //        //new ParametersAndGradients<T>
        //        //{
        //        //    Volume = this.Bias,
        //        //    Gradient = this.BiasGradient,
        //        //    L1DecayMul = Ops<T>.Zero,
        //        //    L2DecayMul = Ops<T>.Zero

        //        //Initialize all properties.
        //        clone.Init(conv.InputWidth, conv.InputHeight, conv.InputDepth);

        //        //Copy all randomized properties.
        //        clone.Filters.Storage.Map(((x,y) => y), conv.Filters.Storage, clone.Filters.Storage);


        //        return clone;
        //    }

        //    var fullConn = source as FullyConnLayer<T>;
        //    if (fullConn != null)
        //    {
        //        fullConn.GetData();
        //        var clone = new FullyConnLayer<T>(fullConn.NeuronCount);

        //        //copy parameters and gradients
        //        //var pg = source.GetParametersAndGradients();
        //        //var clonepg = clone.GetParametersAndGradients();

        //        //for (int i = 0; i < pg.Count; i++)
        //        //{
        //        //    clonepg[i].Volume = BuilderInstance<T>.Volume.SameAs(pg[i].Volume.Storage, pg[i].Volume.Shape);
        //        //    clonepg[i].Gradient = BuilderInstance<T>.Volume.SameAs(pg[i].Gradient.Storage, pg[i].Gradient.Shape);
        //        //    clonepg[i].L1DecayMul = pg[i].L1DecayMul;
        //        //    clonepg[i].L2DecayMul = pg[i].L2DecayMul;
        //        //}

        //        //Initialize all properties.
        //        clone.Init(fullConn.InputWidth, fullConn.InputHeight, fullConn.InputDepth);

        //        //Copy all randomized properties.
        //        clone.Filters.Storage.Map(((x, y) => y), fullConn.Filters.Storage, clone.Filters.Storage);

        //        return clone;
        //    }

        //    var input = source as InputLayer<T>;
        //    if (input != null)
        //        return LayerBase<T>.FromData(input.GetData());
        //    //var layertwin = LayerBase<T>.FromData(layer.GetData()); //Not good enough!!!!

        //    var pool = source as PoolLayer<T>;
        //    if (pool != null)
        //    {
        //        var clone = new PoolLayer<T>(pool.Width, pool.Height);

        //        ////copy parameters and gradients
        //        //var pg = source.GetParametersAndGradients();
        //        //var clonepg = clone.GetParametersAndGradients();

        //        //for (int i = 0; i < pg.Count; i++)
        //        //{
        //        //    clonepg[i].Volume = BuilderInstance<T>.Volume.SameAs(pg[i].Volume.Storage, pg[i].Volume.Shape);
        //        //    clonepg[i].Gradient = BuilderInstance<T>.Volume.SameAs(pg[i].Gradient.Storage, pg[i].Gradient.Shape);
        //        //    clonepg[i].L1DecayMul = pg[i].L1DecayMul;
        //        //    clonepg[i].L2DecayMul = pg[i].L2DecayMul;
        //        //}
        //        return clone;
        //    }

        //    var relu = source as ReluLayer<T>;
        //    if (relu != null)
        //        return LayerBase<T>.FromData(relu.GetData());
        //    //Probably not sufficient. Needs init() to be called. OR NOT????

        //    var sigm = source as SigmoidLayer<T>;
        //    if (sigm != null)
        //        return LayerBase<T>.FromData(sigm.GetData());

        //    var softmax = source as SoftmaxLayer<T>;
        //    if (softmax != null)
        //    {
        //        var clone = new SoftmaxLayer<T>(softmax.ClassCount);

        //        return clone;
        //    }

        //    var tanh = source as SigmoidLayer<T>;
        //    if (tanh != null)
        //        return LayerBase<T>.FromData(tanh.GetData());

        //    throw new Exception($"Copying for {source.GetType().FullName} is not implemented.");
        //}


        public List<ParametersAndGradients<T>> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients<T>>();

            for (int i = 0; i < Layers.Count; i++)
            {
                response.AddRange(this.Layers[i].GetParametersAndGradients());
                response.AddRange(this.LayersTwin[i].GetParametersAndGradients());
            }

            foreach (var t in this.DistanceLayers)
            {
                var parametersAndGradients = t.GetParametersAndGradients();
                response.AddRange(parametersAndGradients);
            }

            return response;
        }

        public T GetCostLoss(Volume<T> input, Volume<T> y)
        {
            throw new NotImplementedException();
        }

        public int[] GetPrediction()
        {

            //Returns forward output for accuracy measurement.
            var lastLayer = DistanceLayers[DistanceLayers.Count - 1];
            var output = lastLayer.OutputActivation.ToArray();
            int[] predictions = new int[output.Length];

            T matchTarget = Ops<T>.Zero;
            T matchThreshold = Ops<T>.Zero;

            //Matching parameters depend on final output.
            var sigmoid = lastLayer as SigmoidLayer<T>;
            if (sigmoid != null)
            {
                matchTarget = Ops<T>.Cast(1.0);
                matchThreshold = Ops<T>.Cast(0.1);
            }

            var joinlayer = lastLayer as ILastLayer<T>;
            if (joinlayer != null)
            {
                matchTarget = Ops<T>.Zero;
                matchThreshold = Ops<T>.Cast(0.5);
            }

            for (int i = 0; i < predictions.Length; i++)
            {
                T upper = Ops<T>.Add(matchTarget, matchThreshold);
                T lower = Ops<T>.Subtract(matchTarget, matchThreshold);

                if (Ops<T>.GreaterThan(upper, output[i]) && Ops<T>.GreaterThan(output[i], lower))
                {
                    predictions[i] = 1;
                }
                else
                {
                    predictions[i] = 0;
                }
            }

            return predictions;

        }

        

        public void Dump(string filename)
        {
            //ADD???
        }

        //Edit to fit
        public static SNet<T> FromData(IDictionary<string, object> dico)
        {
            var snet = new SNet<T>();

            var layers = dico["Layers"] as IEnumerable<IDictionary<string, object>>;
            foreach (var layerData in layers)
            {
                var layer = LayerBase<T>.FromData(layerData);
                var cloneLayer = LayerBase<T>.FromData(layerData);
                snet.Layers.Add(layer);
                snet.LayersTwin.Add(cloneLayer);
            }

            var distLayers = dico["DistLayers"] as IEnumerable<IDictionary<string, object>>;
            foreach (var distLayerData in distLayers)
            {
                var distLayer = LayerBase<T>.FromData(distLayerData);
                snet.DistanceLayers.Add(distLayer);
            }

            return snet;
        }

        public Dictionary<string, object> GetData()
        {
            var dico = new Dictionary<string, object>();
            var layers = new List<Dictionary<string, object>>();
            var distLayers = new List<Dictionary<string, object>>();

            foreach (var layer in this.Layers)
            {
                layers.Add(layer.GetData());
            }
            dico["Layers"] = layers;

            foreach (var distlayer in this.DistanceLayers)
            {
                distLayers.Add(distlayer.GetData());
            }
            dico["DistLayers"] = distLayers;

            return dico;
        }

        public static Volume<T> JoinVolumes(Volume<T> v, Volume<T> v2)
        {
            if (!v.Shape.Equals(v2.Shape)) throw new Exception("Volumes to join must have the same shape.");

            //Join here is 16x size of v or v2.
            //Find last dimension, and double over that.
            //I can just add another dimension.
            //Check last dimension if == joincount.
            //split off of that?
            //Need to find a way to get just a part of the Volume.
            //Volume.toArray() gives a T[], which can be split.
            //Can then rebuild the Volumes with BuilderInstance<T>.Volume.SameAs();

            //Or I can just work with the int[] arrays.
            //Either way, this method is too clunky and not worthwhile...???

            var dim = new List<int>(v.Shape.Dimensions.ToArray());
            dim[dim.Count - 1] *= 2;

            var datasize = v.Shape.TotalLength;
            var data = new T[datasize * 2];
            v.ToArray().CopyTo(data, 0);
            v2.ToArray().CopyTo(data, (int)datasize);

            return BuilderInstance<T>.Volume.SameAs(data, new Shape(dim));
        }

        public static void SplitVolumes(Volume<T> source, out Volume<T> split1, out Volume<T> split2)
        {
            var dim = source.Shape.Dimensions.ToArray();
            //if (dim[dim.Count - 1] != 2) throw new Exception("Volumes to split must be Joined first.");

            //Does it matter if I split by batchsize?????? NOPE.
            dim[dim.Length - 1] /= 2;

            var data = source.ToArray();
            var shape = new Shape(dim);

            var storage = new T[shape.TotalLength];
            var storage2 = new T[shape.TotalLength];

            Array.Copy(data, storage, storage.Length);
            Array.Copy(data, storage.Length, storage2, 0, storage2.Length);

            split1 = BuilderInstance<T>.Volume.SameAs(storage, shape);
            split2 = BuilderInstance<T>.Volume.SameAs(storage2, shape);
        }


    }
}
