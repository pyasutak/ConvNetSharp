using ConvNetSharp.Volume;
using System;
using System.Collections.Generic;
using System.Text;

namespace ConvNetSharp.SNet.Layers
{
    interface IJoinLayer
    {
        int JoinCount { get; }

        Volume<double> DoJoin(bool isTraining = false, params Volume<double>[] inputs);
    }
}
