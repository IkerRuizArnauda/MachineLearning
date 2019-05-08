// Iker Ruiz Arnauda 2019
using System;
using System.IO;

using Testonnxruntime.OnnxWrap;

namespace Testonnxruntime
{
    public class SampleSqueezenet
    {
        public SampleSqueezenet()
        {
            try
            {
                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "squeezenet.onnx");
                var modelLabels = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "SqueezenetLabels.txt");
                using (var OnnxSesssion = new OnnxSession(modelPath, modelLabels))
                {
                    var filePath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "bench.in");
                    float[] data = DataHelper.LoadTensorFromFile(filePath);

                    if (OnnxSesssion.Infere(new float[][] { data }, out Results results))
                        results.PrintPrediction();
                    else
                        Console.WriteLine("Unable to infere model.");
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
            }
            finally
            {
                Console.ReadLine();
            }
        }
    }
}
