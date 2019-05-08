// Iker Ruiz Arnauda 2019

using System;
using System.Collections.Generic;

namespace Testonnxruntime.OnnxWrap
{
    public class Results
    {
        public int Prediction { get; set; }
        public string[] Labels;
        public List<double> InferenceResults { get; set; }

        public void PrintPrediction()
        {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine(GetPrediction());
            Console.ResetColor();
        }

        private string GetPredictionLabel()
        {
            if (Labels != null && Prediction < Labels.Length)
                return $"Label: {Labels[Prediction]}";
            else
                return "Label: N/A";
        }

        public string GetAllScores()
        {
            return $"Scores:\n{string.Join("\n", InferenceResults)}";
        }

        public string GetPrediction()
        {
            return $"------------------------------------------\nNeural Network Result:\nPrediction: {Prediction} {GetPredictionLabel()}\n------------------------------------------";
        }
    }
}
