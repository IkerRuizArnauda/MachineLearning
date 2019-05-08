// Iker Ruiz Arnauda 2019
using System;
using System.IO;
using System.Drawing;
using System.Collections.Generic;

using Testonnxruntime.OnnxWrap;

namespace Testonnxruntime
{
    public class SampleSqueezenet2
    {
        public SampleSqueezenet2()
        {
            try
            {
                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "squeezenet.onnx");
                var modelLabels = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "SqueezenetLabels.txt");
                using (var OnnxSesssion = new OnnxSession(modelPath, modelLabels))
                {
                    var bmapPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "dog2.jpg");
                    using (Bitmap bitmap = (Bitmap)Image.FromFile(bmapPath, false))
                    {
                        List<float[]> data = new List<float[]>();
                        foreach (var input in OnnxSesssion.InputShapes)
                            data.Add(DataHelper.GetBitmapFloatArray(bitmap, OnnxSesssion, input.Key, false));

                        if (OnnxSesssion.Infere(data.ToArray(), out Results results))
                            results.PrintPrediction();
                        else
                            Console.WriteLine("Unable to infere model.");
                    }
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
