using System;
using System.Collections.Generic;
using System.Text;
using ConvNetSharp.Core;
using ConvNetSharp.Core.Layers;
using ConvNetSharp.Volume;
using ConvNetSharp.SNet;
using ConvNetSharp.Core.Serialization;

namespace ConvNetSharp.SNet.Layers
{
    public class TwinJoinLayer : LayerBase<double>, IJoinLayer
    {

        public TwinJoinLayer(Dictionary<string, object> dico) : base(dico)
        {
            this.Alpha = BuilderInstance<double>.Volume.SameAs(dico["Alpha"].ToArrayOfT<double>(), new Shape(1, 1, this.InputWidth * this.InputHeight * this.InputDepth));

            this.IsInitialized = true;
        }

        public TwinJoinLayer() : base() { }


        public int JoinCount { get; } = 2;

        public Volume<double> InputTwinActivation { get; protected set; }

        public Volume<double> InputTwinActivationGradients { get; protected set; }

        public Volume<double> Alpha { get; protected set; }

        public Volume<double> AlphaGradients { get; protected set; }


        public override void Backward(Volume<double> outputGradient)
        {

            this.OutputActivationGradients = outputGradient;

            //var rs = InputActivation.ReShape(1, 1, -1, InputActivation.Shape.GetDimension(3)).Storage; //Required???
            //var rsB = InputTwinActivation.ReShape(1, 1, -1, InputTwinActivation.Shape.GetDimension(3)).Storage;
            //SVolume join = new SVolume(rs, rsB);

            SVolume join = new SVolume(InputActivation.Storage, InputTwinActivation.Storage);

            join.DoNetworkJoinGradients(
                this.Alpha, this.OutputActivationGradients,
                this.InputActivationGradients, this.InputTwinActivationGradients, this.AlphaGradients);


            //// compute gradient wrt weights and data
            //using (var reshapedInput = this.InputActivation.ReShape(1, 1, -1, this.InputActivation.Shape.GetDimension(3)))
            //using (var reshapedInputGradients = this.InputActivationGradients.ReShape(1, 1, -1, this.InputActivationGradients.Shape.GetDimension(3)))
            //{
            //    reshapedInput.ConvolveGradient(
            //        this.Filters, this.OutputActivationGradients,
            //        reshapedInputGradients, this.FiltersGradient,
            //        0, 1);

            //    this.OutputActivationGradients.BiasGradient(this.BiasGradient);
            //}
        }

        public override Volume<double> DoForward(Volume<double> input, bool isTraining = false)
        {
#if DEBUG
            var inputs = input.ToArray();
            foreach (var i in inputs)
                if (Ops<double>.IsInvalid(i))
                    throw new ArgumentException("Invalid input!");
#endif


            //Need to split the input.



            this.InputActivation = input;
            //this.InputTwinActivation = input2;


            //Shape needs to be re examined after input is split.
            var outputShape = new Shape(this.OutputWidth, this.OutputHeight, this.OutputDepth, input.Shape.DimensionCount == 4 ? input.Shape.GetDimension(3) : 1);

            if (this.OutputActivation == null ||
                !this.OutputActivation.Shape.Equals(outputShape))
            {
                this.OutputActivation = BuilderInstance<double>.Volume.SameAs(input.Storage, outputShape);
            }

            if (this.InputActivationGradients == null ||
                !this.InputActivationGradients.Shape.Equals(input.Shape))
            {
                this.InputActivationGradients = BuilderInstance<double>.Volume.SameAs(this.InputActivation.Storage,
                    this.InputActivation.Shape);
            }
            if (this.InputTwinActivationGradients == null ||
              !this.InputTwinActivationGradients.Shape.Equals(input.Shape))
            {
                this.InputTwinActivationGradients = BuilderInstance<double>.Volume.SameAs(this.InputActivation.Storage,
                    this.InputActivation.Shape);
            }

            this.OutputActivation = Forward(input, isTraining);

            throw new NotImplementedException();
            //return this.OutputActivation;
        }

        //Should probably rename this method to distinguish between the Layer dofoward and the IJoinLayer doForward.
        public virtual Volume<double> DoForward(bool isTraining = false, params Volume<double>[] inputs)
        {
#if DEBUG
            foreach (var input in inputs)
            {
                var list = input.ToArray();
                foreach (var i in list)
                    if (Ops<double>.IsInvalid(i))
                        throw new ArgumentException("Invalid input!");
            }
#endif
            if (inputs.Length != this.JoinCount)
                throw new ArgumentException($"Invalid number of inputs! Should have {this.JoinCount} inputs.");

            this.InputActivation = inputs[0];
            this.InputTwinActivation = inputs[1];

            var outputShape = new Shape(this.OutputWidth, this.OutputHeight, this.OutputDepth, inputs[0].Shape.DimensionCount == 4 ? inputs[0].Shape.GetDimension(3) : 1);

            if (this.OutputActivation == null ||
                !this.OutputActivation.Shape.Equals(outputShape))
            {
                this.OutputActivation = BuilderInstance<double>.Volume.SameAs(inputs[0].Storage, outputShape);
            }

            if (this.InputActivationGradients == null ||
                !this.InputActivationGradients.Shape.Equals(inputs[0].Shape))
            {
                this.InputActivationGradients = BuilderInstance<double>.Volume.SameAs(this.InputActivation.Storage,
                    this.InputActivation.Shape);
            }

            if (this.InputTwinActivationGradients == null ||
              !this.InputTwinActivationGradients.Shape.Equals(inputs[1].Shape))
            {
                this.InputTwinActivationGradients = BuilderInstance<double>.Volume.SameAs(this.InputActivation.Storage,
                    this.InputActivation.Shape);
            }

            this.OutputActivation = Forward(inputs[0], inputs[1], isTraining);

            return this.OutputActivation;
        }

        protected override Volume<double> Forward(Volume<double> input, bool isTraining = false)
        {
            /*
             * If single input, assume it is both required input as one. So Split the volume by JoinCount.
             */

            //Might need to cast input as a SVolume.

            throw new NotImplementedException();
        }

        protected virtual Volume<double> Forward(Volume<double> input, Volume<double> inputB, bool isTraining = false)
        {

            //var rs = input.ReShape(1, 1, -1, input.Shape.GetDimension(3)).Storage; //Is this necessary??? Mostly copying from FullyConnLayer
            //var rsB = inputB.ReShape(1, 1, -1, input.Shape.GetDimension(3)).Storage;
            //SVolume join = new SVolume(rs, rsB);

            //Let's try this...
            SVolume join = new SVolume(input.Storage, inputB.Storage);

            join.DoNetworkJoin(this.Alpha, this.OutputActivation);
            return this.OutputActivation;


            //reshapedInput.DoConvolution(this.Filters, 0, 1, this.OutputActivation);
            //this.OutputActivation.DoAdd(this.Bias, this.OutputActivation);
            //return this.OutputActivation;
        }

        public override List<ParametersAndGradients<double>> GetParametersAndGradients()
        {
            var response = new List<ParametersAndGradients<double>>
            {
                new ParametersAndGradients<double>
                {
                    Volume = this.Alpha,
                    Gradient = this.AlphaGradients,
                    L1DecayMul = Ops<double>.Zero,
                    L2DecayMul = Ops<double>.Zero
                }
            };

            return response;
        }

        public override Dictionary<string, object> GetData()
        {
            Dictionary<string, object> dict = base.GetData();

            dict["Alpha"] = this.Alpha.ToArray();

            return dict;
        }

        public override void Init(int inputWidth, int inputHeight, int inputDepth)
        {
            base.Init(inputWidth, inputHeight, inputDepth);
            this.OutputWidth = 1;
            this.OutputHeight = 1;
            this.OutputDepth = 1;

            var inputCount = this.InputWidth * this.InputHeight * this.InputDepth;

            this.Alpha = BuilderInstance<double>.Volume.Random(new Shape(1, 1, inputCount)); // Randomization needs to be optimized?
            this.AlphaGradients = BuilderInstance<double>.Volume.SameAs(new Shape(1, 1, inputCount));
        }

    }
}
