﻿using System;
using System.Collections.Generic;
using ConvNetSharp.Volume;

namespace ConvNetSharp.Core.Layers
{
    public class SigmoidLayer<T> : LayerBase<T>, ILastLayer<T> where T : struct, IEquatable<T>, IFormattable
    {
        public SigmoidLayer(Dictionary<string, object> data) : base(data)
        {
        }

        public SigmoidLayer()
        {
        }

        public override void Backward(Volume<T> outputGradient)
        {
            this.OutputActivationGradients = outputGradient;
            this.OutputActivation.DoSigmoidGradient(this.InputActivation, this.OutputActivationGradients, this.InputActivationGradients);
        }
        
        public void Backward(Volume<T> y, out T loss)
        {
            // input gradient = pi - yi
            this.OutputActivationGradients = BuilderInstance<T>.Volume.SameAs(y.Shape);
            y.DoSubtractFrom(this.OutputActivation, this.OutputActivationGradients);
            
            this.OutputActivation.DoSigmoidGradient(this.InputActivation, this.OutputActivationGradients, this.InputActivationGradients);

            //Cross-Entropy Loss
            loss = Ops<T>.Zero;
            for (var N = 0; N < y.Shape.GetDimension(3); N++)
            {
                for (var d = 0; d < y.Shape.GetDimension(2); d++)
                {
                    for (var h = 0; h < y.Shape.GetDimension(1); h++) //always 1
                    {
                        for (var w = 0; w < y.Shape.GetDimension(0); w++) //always 1
                        {
                            var expected = y.Get(w, h, d, N); //either 1 (match) or 0 (nomatch)
                            var euclideanDistance = this.OutputActivation.Get(w, h, d, N);

                            var match   = Ops<T>.Multiply(expected, Ops<T>.Log(euclideanDistance));
                            var nomatch = Ops<T>.Multiply(Ops<T>.Subtract(Ops<T>.One, expected), Ops<T>.Log(Ops<T>.Subtract(Ops<T>.One, euclideanDistance)));

                            var current = Ops<T>.Add(match, nomatch);
                            loss = Ops<T>.Add(loss, current);
                        }
                    }
                }
            }
            loss = Ops<T>.Negate(loss); //Optimization
        }

        protected override Volume<T> Forward(Volume<T> input, bool isTraining = false)
        {
            input.DoSigmoid(this.OutputActivation);
            return this.OutputActivation;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);

            this.OutputDepth = inputDepth;
            this.OutputWidth = inputWidth;
            this.OutputHeight = inputHeight;
        }

    }
}