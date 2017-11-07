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
        public List<LayerBase<T>> Layers { get; } = new List<LayerBase<T>>();
        public List<LayerBase<T>> LayersTwin { get; } = new List<LayerBase<T>>();

        public List<LayerBase<T>> DistanceLayers { get; } = new List<LayerBase<T>>();


        public Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            SplitVolumes(input, out Volume<T> split1, out Volume<T> split2);
            return this.Forward(split1, split2, isTraining);
        }

        public Volume<T> Forward(Volume<T>[] input, bool isTraining = false)
        {
            return this.Forward(input[0], input[1], isTraining);
        }

        public Volume<T> Forward(Volume<T> inputA, Volume<T> inputB, bool isTraining = false)
        {
            //try
            //{
            //    ValidateTwin();
            //}
            //catch
            //{
            //    BalanceParameters();
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

            var distance = joinLayer.DoJoin(isTraining, outA, outB) as Volume<T>; //Urgh, the generics....

            for (var i = 1; i < this.DistanceLayers.Count; i++)
            {
                var distanceLayer = this.DistanceLayers[i];
                distance = distanceLayer.DoForward(distance, isTraining);
            }

            return distance;
        }

        public T Backward(Volume<T> y)
        {
            var n = this.Layers.Count;
            var dn = this.DistanceLayers.Count;
            var lastLayer = this.Layers[n - 1];
            var lastLayerTwin = this.LayersTwin[n - 1];
            var lastDistanceLayer = this.DistanceLayers[dn - 1] as ILastLayer<T>;
            if (lastLayer != null && lastLayerTwin != null)
            {

                // Calculate Gradients for Distance Layer
                lastDistanceLayer.Backward(y, out T loss);
                for (int i = dn - 2; i >= 0; i--)
                {
                    this.DistanceLayers[i].Backward(this.DistanceLayers[i + 1].InputActivationGradients);
                }
                
                // Split SVolume into two Volumes for standard processing in the Twin Layers.
                var distanceLayerTwinGradients = (this.DistanceLayers[0] as TwinJoinLayer).InputTwinActivationGradients as Volume<T>;

                // Calculate Gradients for Twin Layers
                lastLayer.Backward(this.DistanceLayers[0].InputActivationGradients);
                lastLayerTwin.Backward(distanceLayerTwinGradients);
                this.BalanceGradients(n - 1); //Balance Gradients so twin layers parameters remain the same.
                for (int i = n - 2; i >= 0; i--)
                {
                    // first layer assumed input 
                    this.Layers[i].Backward(this.Layers[i + 1].InputActivationGradients);
                    this.LayersTwin[i].Backward(this.LayersTwin[i + 1].InputActivationGradients);

                    //Balance Gradients
                    this.BalanceGradients(i);
                }

                
                //Calculate Loss. --------------> Moved to lastDistanceLayer
                //var y = y;

                //for (var N = 0; N < y.Shape.GetDimension(3); N++)
                //{
                //    for (var d = 0; d < y.Shape.GetDimension(2); d++)
                //    {
                //        for (var h = 0; h < y.Shape.GetDimension(1); h++)
                //        {
                //            for (var w = 0; w < y.Shape.GetDimension(0); w++)
                //            {
                //                var expected = y.Get(w, h, d, N);
                //                var actual = lastDistanceLayer.OutputActivation.Storage.Get(w, h, d, N);
                //                if (Ops<T>.Zero.Equals(actual))
                //                    actual = Ops<T>.Epsilon;
                //                var current = Ops<T>.Multiply(expected, Ops<T>.Log(actual));

                //                loss = Ops<T>.Add(loss, current);
                //            }
                //        }
                //    }
                //}
                //loss = Ops<T>.Negate(loss);
                
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
        
        //Currently unused. Required for testing.
        //public void ValidateTwin()
        //{
        //    Func<T, T, T> verify = (x, y) => Ops<T>.Equals(x, y) ? x : throw new Exception("The Siamese Net is invalid.");

        //    for (int i = 0; i < Layers.Count; i++)
        //    {
        //        var l = Layers[i];
        //        var lt = LayersTwin[i];
        
        //        var convl = l as ConvLayer<T>;
        //        if (convl != null)
        //        {
        //            var convlt = lt as ConvLayer<T>;
        //            if (convlt == null) throw new Exception("The Layers in the Siamese Net are not valid!");

        //            var filtersVerification = BuilderInstance<T>.Volume.SameAs(convl.Filters.Shape).Storage;
        //            convl.Filters.Storage.Map(verify, convlt.Filters.Storage, filtersVerification);
        //            var biasVerification = BuilderInstance<T>.Volume.SameAs(convl.Bias.Shape).Storage;
        //            convl.Bias.Storage.Map(verify, convlt.Bias.Storage, biasVerification);
        //        }

        //        var connl = l as FullyConnLayer<T>;
        //        if (connl != null)
        //        {
        //            var connlt = lt as FullyConnLayer<T>;
        //            if (connlt == null) throw new Exception("The Layers in the Siamese Net are not valid!");

        //            var filtersVerification = BuilderInstance<T>.Volume.SameAs(connl.Filters.Shape).Storage;
        //            connl.Filters.Storage.Map(verify, connlt.Filters.Storage, filtersVerification);
        //            var biasVerification = BuilderInstance<T>.Volume.SameAs(connl.Bias.Shape).Storage;
        //            connl.Bias.Storage.Map(verify, connlt.Bias.Storage, biasVerification);
        //        }

        //        var pl = l as PoolLayer<T>;
        //        if (pl != null)
        //        {
        //            var plt = lt as PoolLayer<T>;
        //            if (plt == null) throw new Exception("The Layers in the Siamese Net are not valid!");

        //            if (pl.Width != plt.Width) throw new Exception("Poll Layer's Widths do not match.");
        //            if (pl.Height != plt.Height) throw new Exception("Poll Layer's Heights do not match.");
        //            if (pl.Stride != plt.Stride) throw new Exception("Poll Layer's Strides do not match.");
        //            if (pl.Pad != plt.Pad) throw new Exception("Poll Layer's Pads do not match.");
        //        }

        //    }
        //}


        private void BalanceGradients(int layerIndex)
        {
            //updated now to use GetParametersAndGradients. Works as intended.

            var pandg = Layers[layerIndex].GetParametersAndGradients();
            var pandgT = LayersTwin[layerIndex].GetParametersAndGradients();

            for (int i = 0; i < pandg.Count; i++)
            {
                var gradient = pandg[i].Gradient;
                var gradientT = pandgT[i].Gradient;
                var temp = gradient.Clone();
                
                //var avg = BuilderInstance<T>.Volume.SameAs(gradient.Shape);
                //gradient.Storage.Map((x, y) => Ops<T>.Divide(Ops<T>.Add(x, y), Ops<T>.Cast(2)), gradientT.Storage, avg.Storage);

                //var avg = gradient + gradientT * Ops<T>.Cast(0.5);

                gradient.DoAdd(gradientT, gradient);
                //gradient.DoMultiply(gradient, Ops<T>.Cast(0.5)); //Additive instead of average
                gradientT.DoAdd(temp, gradientT);
                //gradientT.DoMultiply(gradientT, Ops<T>.Cast(0.5));
                temp.Dispose();
            }
        }


        //made obsolete by BalanceGradients()
        //private void BalanceParameters()
        //{
        //    Func<T, T, T> avg = (x, y) => Ops<T>.Divide(Ops<T>.Add(x, y), Ops<T>.Cast(2.0));
        //    //Func<T, T> copy = x => x;
        
        //    //Func<T, T, T> verify = (x, y) => Ops<T>.Equals(x, y) ? x : throw new Exception("The Siamese Net is invalid.");
        
        //    for (int i = 0; i < this.Layers.Count; i++)
        //    {
        //        var layerParams = this.Layers[i].GetParametersAndGradients();
        //        var layerTwinParams = this.LayersTwin[i].GetParametersAndGradients();

        //        for (int j = 0; j < layerParams.Count; j++)
        //        {
        //            //Average Volume.
        //            Volume<T> average = BuilderInstance<T>.Volume.SameAs(layerParams[j].Volume.Shape);
        //            layerParams[j].Volume.Storage.Map(avg, layerTwinParams[j].Volume.Storage, average.Storage);
        //            layerParams[j].Volume.Storage.CopyFrom(average.Storage);
        //            layerTwinParams[j].Volume.Storage.CopyFrom(average.Storage);
        //            //Average Gradient.
        //            average = BuilderInstance<T>.Volume.SameAs(layerParams[j].Gradient.Shape);
        //            layerParams[j].Gradient.Storage.Map(avg, layerTwinParams[j].Volume.Storage, average.Storage);
        //            layerParams[j].Gradient.Storage.CopyFrom(average.Storage);
        //            layerTwinParams[j].Gradient.Storage.CopyFrom(average.Storage);
        //        }

        //    }

        //}


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

            if (layer is IClassificationLayer classificationLayer)
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

            if (layer is ReluLayer<T> reluLayer)
            {
                if (lastLayer is IDotProductLayer<T> dotProductLayer)
                {
                    // relus like a bit of positive bias to get gradients early
                    // otherwise it's technically possible that a relu unit will never turn on (by chance)
                    // and will never get any gradient and never contribute any computation. Dead relu.

                    dotProductLayer.BiasPref = (T)Convert.ChangeType(0.1, typeof(T)); // can we do better?
                    //Update Twin.
                    var twinDotProductLayer = this.LayersTwin[this.Layers.Count - 1] as IDotProductLayer<T>;
                    twinDotProductLayer.BiasPref = dotProductLayer.BiasPref;
                    //Both ConvLayer and FullyConnLayer have filters that are updated by this. Copy over to twin.

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

            if (this.Layers.Count > 0)
            {
                layer.Init(inputWidth, inputHeight, inputDepth);
            }

            //Clone the layer.
            var layertwin = LayerBase<T>.FromData(layer.GetData());

            this.Layers.Add(layer);
            this.LayersTwin.Add(layertwin);
        }

        //Should only be used after finished adding normal net layers.
        public void AddDistanceLayer(LayerBase<T> dlayer)
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
            else if (!(dlayer is IJoinLayer))
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
            
            dlayer.Init(inputWidth, inputHeight, inputDepth);

            this.DistanceLayers.Add(dlayer);
        }
        
        //Used to get references to trainable values (weights and other parameters) and the gradients to change them.
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

        //Used for testing? Copied direct from Core.Net
        public T GetCostLoss(Volume<T> input, Volume<T> y)
        {

            SplitVolumes(input, out Volume<T> split1, out Volume<T> split2);
            if (!split1.Equals(this.Layers[0].InputActivation) || !split2.Equals(this.LayersTwin[0].InputActivation) )
            {
                Forward(split1,split2);
            }

            if (this.DistanceLayers[this.DistanceLayers.Count - 1] is ILastLayer<T> lastLayer)
            {
                lastLayer.Backward(y, out T loss);
                return loss;
            }

            throw new Exception("Last layer doesn't implement ILastLayer interface");
        }
        
        //Used for accuracy measurement.
        public int[] GetPrediction()
        {
            //Returns forward output for accuracy measurement.
            //var lastLayer = DistanceLayers[DistanceLayers.Count - 1];
            //var output = lastLayer.OutputActivation.ToArray();
            var output = DistanceLayers[DistanceLayers.Count - 1].OutputActivation.ToArray();
            int[] predictions = new int[output.Length];

            //T matchTarget = Ops<T>.Zero;
            //T matchThreshold = Ops<T>.Zero;

            T matchTarget = Ops<T>.Cast(1.0);
            T matchThreshold = Ops<T>.Cast(0.2);

            ////Matching parameters depend on final output.
            //var sigmoid = lastLayer as SigmoidLayer<T>;
            //if (sigmoid != null)
            //{
            //    matchTarget = Ops<T>.Cast(1.0);
            //    matchThreshold = Ops<T>.Cast(0.2);
            //}

            //var joinlayer = lastLayer as IJoinLayer;
            //if (joinlayer != null)
            //{
            //    matchTarget = Ops<T>.Zero;
            //    matchThreshold = Ops<T>.Cast(0.5);
            //}

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

        //Save net data to a text file... I believe it was an early testing requirement...
        public void Dump(string filename)
        {
            //ADD???
        }

        //Build an SNet from dictionary provided by this.GetData()
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

        //Provides all information required to recreate SNet.
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
        
        //SVolume utility function.
        public static Volume<T> JoinVolumes(Volume<T> v, Volume<T> v2)
        {
            if (!v.Shape.Equals(v2.Shape)) throw new Exception("Volumes to join must have the same shape.");

            var dim = new List<int>(v.Shape.Dimensions.ToArray());
            dim[dim.Count - 1] *= 2;

            var datasize = v.Shape.TotalLength;
            var data = new T[datasize * 2];
            v.ToArray().CopyTo(data, 0);
            v2.ToArray().CopyTo(data, (int)datasize);

            return BuilderInstance<T>.Volume.SameAs(data, new Shape(dim));
        }

        //SVolume utility function.
        public static void SplitVolumes(Volume<T> source, out Volume<T> split1, out Volume<T> split2)
        {
            var dim = source.Shape.Dimensions.ToArray();
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
