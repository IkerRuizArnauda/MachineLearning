// Iker Ruiz Arnauda 2019

using System;
using System.IO;
using System.Linq;
using System.Numerics.Tensors;
using System.Collections.Generic;

using Microsoft.ML.OnnxRuntime;

namespace Testonnxruntime.OnnxWrap
{
    public class OnnxSession : IDisposable
    {
        public InferenceSession _session { get; set; }
        public Dictionary<string, int[]> InputShapes = new Dictionary<string, int[]>();
        public Dictionary<string, int[]> OutputShapes = new Dictionary<string, int[]>();
        public string[] Labels;

        public OnnxSession(string modelPath, string labelsPath = "")
        {
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine("Loading model session...\n" + modelPath);
            Console.ResetColor();

            _session = new InferenceSession(modelPath);

            Console.WriteLine("Loading model inputs metadata...");

            foreach (var input in _session.InputMetadata)
            {
                if (!input.Value.IsTensor)
                    continue;

                input.PrintInformation(true);

                if (!InputShapes.ContainsKey(input.Key))
                    InputShapes.Add(input.Key, null);

                InputShapes[input.Key] = input.Value.Dimensions;
            }

            Console.WriteLine("Loading model outputs metadata...");
            foreach (var output in _session.OutputMetadata)
            {
                if (!output.Value.IsTensor)
                    continue;

                output.PrintInformation(false);

                if (!OutputShapes.ContainsKey(output.Key))
                    OutputShapes.Add(output.Key, null);

                OutputShapes[output.Key] = output.Value.Dimensions;
            }

            if (!string.IsNullOrEmpty(labelsPath))
            {
                if (File.Exists(labelsPath))
                    using (StreamReader sw = new StreamReader(labelsPath))
                        Labels = sw.ReadToEnd().Split('\n');
            }
        }

        public int GetBatchSize(string key)
        {
            return InputShapes[key][1];
        }

        public int GetShapeWIdth(string key)
        {
            return InputShapes[key][2];
        }

        public int GetShapeHeight(string key)
        {
            return InputShapes[key][3];
        }

        private List<NamedOnnxValue> BuildInputContainers(ICollection<float[]> floatTensors)
        {
            var data = floatTensors.ToArray();

            if (data.Length != InputShapes.Keys.Count)
                throw new Exception($"Expecting data for {InputShapes.Keys.Count} shapes, received {data.Length}.");

            var container = new List<NamedOnnxValue>();
            var inputIndex = 0;

            foreach (var input in InputShapes.Keys)
            {
                var tensor = new DenseTensor<float>(data[inputIndex], InputShapes[input]);
                container.Add(NamedOnnxValue.CreateFromTensor<float>(input, tensor));
                inputIndex++;
            }

            return container;
        }

        public bool Infere(float[][] data, out Results results)
        {
            results = null;
            if (_session == null)
                return false;

            try
            {
                var container = BuildInputContainers(data);
                var inference = _session.Run(container).ToArray();
                var inferenceResults = inference[0];
                var prediction = inferenceResults.AsTensor<float>().ToArray();

                results = new Results()
                {
                    Labels = Labels?.ToArray(),
                    Prediction = Array.IndexOf(prediction, prediction.Max()),
                    InferenceResults = prediction.Select(i => (double)i).ToList(),
                };

                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return false;
            }
        }

        private bool disposed = false;
        protected virtual void Dispose(bool disposing)
        {
            if (!disposed)
            {
                if (disposing)
                {
                    InputShapes?.Clear();
                    OutputShapes?.Clear();
                    _session?.Dispose();
                }

                disposed = true;
            }
        }

        public void Dispose()
        {
            Dispose(true);
        }
    }
}
