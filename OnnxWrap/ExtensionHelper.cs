// Iker Ruiz Arnauda 2019

using System;
using System.Collections.Generic;

using Microsoft.ML.OnnxRuntime;

namespace Testonnxruntime.OnnxWrap
{
    public static class ExtensionHelper
    {
        public static void PrintInformation(this KeyValuePair<string, NodeMetadata> data, bool inputs)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;

            if (inputs)
                Console.WriteLine("Model Input");
            else
                Console.WriteLine("Model Output");

            Console.WriteLine("Name: {0}", data.Key);
            Console.WriteLine("Shape: [{0}]", string.Join(",", data.Value.Dimensions));
            Console.WriteLine("Type: {0}", data.Value.ElementType.Name);
            Console.WriteLine("IsTensor: {0}", data.Value.IsTensor);
            Console.ResetColor();
        }
    }
}
