// Iker Ruiz Arnauda 2019
using System;
using System.IO;
using System.Drawing;
using System.Collections.Generic;

using Testonnxruntime.OnnxWrap;

namespace Testonnxruntime
{
    public class SampleIdentifyDigit
    {
        public SampleIdentifyDigit()
        {
            try
            {
                var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "model.onnx");
                using (var OnnxSession = new OnnxSession(modelPath))
                {
                    using (Bitmap bitmap = (Bitmap)Bitmap.FromFile(Path.Combine(AppDomain.CurrentDomain.BaseDirectory + @"Resources\", "seven.bmp")))
                    {
                        List<float[]> data = new List<float[]>();
                        foreach (var input in OnnxSession.InputShapes)
                            data.Add(DataHelper.GetBitmapFloatArray(bitmap, OnnxSession, input.Key, false));

                        if (OnnxSession.Infere(data.ToArray(), out Results results))
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
