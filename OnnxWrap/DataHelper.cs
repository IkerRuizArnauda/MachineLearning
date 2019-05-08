// Iker Ruiz Arnauda 2019

using System;
using System.IO;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Collections.Generic;
using Testonnxruntime.ImageVisualizer;

namespace Testonnxruntime.OnnxWrap
{
    public static class DataHelper
    {
        public static Bitmap ResizeImage(Image image, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destImage = new Bitmap(width, height);

            destImage.SetResolution(image.HorizontalResolution, image.VerticalResolution);

            using (var graphics = Graphics.FromImage(destImage))
            {
                graphics.CompositingMode = CompositingMode.SourceCopy;
                graphics.CompositingQuality = CompositingQuality.HighQuality;
                graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = PixelOffsetMode.HighQuality;

                using (var wrapMode = new ImageAttributes())
                {
                    wrapMode.SetWrapMode(WrapMode.TileFlipXY);
                    graphics.DrawImage(image, destRect, 0, 0, image.Width, image.Height, GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destImage;
        }

        public static float[] GetBitmapFloatArray(Bitmap original, OnnxSession session, string inputKey, bool bw)
        {
            var batchSize = session.GetBatchSize(inputKey);
            var shapeWidth = session.GetShapeWIdth(inputKey);
            var shapeHeight = session.GetShapeHeight(inputKey);

            using (original)
            using (var resized = ResizeImage(original, shapeWidth, shapeHeight))
            {
                Console.ForegroundColor = ConsoleColor.White;
                Console.WriteLine("Processing image:");
                Console.ResetColor();
                ConsoleImage.ConsoleWriteImage(resized);
                float[] data = new float[batchSize * (shapeWidth * shapeHeight)];
                for (int x = 0; x < resized.Width; x++)
                {
                    for (int y = 0; y < resized.Height; y++)
                    {
                        var color = resized.GetPixel(x, y);

                        float pixelValue = color.ToArgb();

                        if (bw)
                            pixelValue = 255 - (color.R + color.G + color.B) / 3;

                        //Todo, how do we handle batchsize > 1?, Fr now just grow the array cloning the first batch.
                        for (int i = 0; i < batchSize; i++)
                        {
                            var pos = y * resized.Width + x + (shapeWidth * shapeWidth * i);
                            data[pos] = pixelValue;
                        }
                    }
                }

                return data;
            }
        }

        public static float[] LoadTensorFromFile(string filename)
        {
            var tensorData = new List<float>();

            try
            {
                using (var inputFile = new StreamReader(filename))
                {
                    inputFile.ReadLine();
                    string[] dataStr = inputFile.ReadLine().Split(new char[] { ',', '[', ']' }, StringSplitOptions.RemoveEmptyEntries);
                    for (int i = 0; i < dataStr.Length; i++)
                        tensorData.Add(Single.Parse(dataStr[i]));
                }

                return tensorData.ToArray();
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.Message);
                return tensorData.ToArray();
            }
        }
    }
}
