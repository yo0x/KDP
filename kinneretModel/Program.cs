using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using PLplot;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using static Microsoft.ML.Transforms.Normalizers.NormalizingEstimator;
using Regression_AreaCalcPredictions.DataStructures;
using Common;
using Microsoft.ML.Data;

//floor1,floor2 ,floor3,floor4,temperature,area

namespace kinneretModel
{
    
    
    internal static class Program
    {
        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string BaseDatasetsLocation = @"../../../../Data";
        private static string TrainDataPath = $"../MOCK_DATA-train.csv";
        private static string TestDataPath = $"../MOCK_DATA-test.csv";
        private static string BaseModelsPath = @"../MLModels";
        private static string ModelPath = $"{BaseModelsPath}/kinneretModel.zip";
        static void Main(string[] args)
        {
            
            MLContext mlContext = new MLContext(seed: 0);
            BuildTrainEvaluateAndSaveModel(mlContext);
            TestSinglePrediction(mlContext);
            PlotRegressionChart(mlContext, TestDataPath, 100,args);
            Console.WriteLine("Press any key to exit..");
            Console.ReadLine();

           
        }
        private static ITransformer BuildTrainEvaluateAndSaveModel(MLContext mLContext)
        {
             //Loading the data from csv
            TextLoader textLoader = mlContext.Data.CreateTextReader(new[]
                                                                    {
                                                                        new TextLoader.Column("floor1", DataKind.R4,0),
                                                                        new TextLoader.Column("floor2", DataKind.R4,1),
                                                                        new TextLoader.Column("floor3", DataKind.R4,2),
                                                                        new TextLoader.Column("floor4", DataKind.R4,3),
                                                                        new TextLoader.Column("temperature", DataKind.R4,4),
                                                                        new TextLoader.Column("area", DataKind.R4,5)
                                                                    },
                                                                    hasHeader: true,
                                                                    separatorChar: ','
                                                                        
                                                                    );
            IDataView baseTrainingDataView = textLoader.Read(TrainDataPath);
            IDataView testDataView = textLoader.Read(TestDataPath);
            //Extreme data remover, to take onle fair input
            var cnt = baseTrainingDataView.GetColumn<float>(mlContext,"area").Count();
            IDataView trainingDataView = mlContext.Data.FilterByColumn(baseTrainingDataView,"area",lowerBound:1, upperBound:150);
            var cnt2 = trainingDataView.GetColumn<float>(mlContext, "area").Count();
            //Data process confg
            var dataProcessPipeline = mlContext.Transforms.CopyColumns("area", "Label")
                            .Append(mlContext.Transforms.Normalize(inputName:"floor1","floor1Encoded"))
                            .Append(mlContext.Transforms.Normalize(inputName:"floor2","floor2Encoded"))
                            .Append(mlContext.Transforms.Normalize(inputName:"floor3","floor3Encoded"))
                            .Append(mlContext.Transforms.Normalize(inputName:"floor4","floor4Encoded"))
                            .Append(mlContext.Transforms.Normalize(inputName:"temperature","temperatureEncoded"))
                            .Append(mlContext.Transforms.Concatenate("Features","floor1Encoded","floor2Encoded","floor3Encoded","floor4Encoded","temperatureEncoded"));

            var trainer = mlContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumn:"Label",featureColumn:"Features");
            var trainingPipeline = dataProcessPipeline.Append(trainer);
//step 4 Trining the model
            var trainedModel = trainingPipeline.Fit(trainingDataView);
// step 5 Evalute the model against TEst Data.
            IDataView predictions = trainedModel.Transform(testDataView);
            var metrics = mlContext.Regression.Evaluate(predictions, label:"Label",score:"Score");
            
            Common.ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);
            //saving model to ZIP
            using (var fs = File.Create(ModelPath))
                trainedModel.SaveTo(mlContext, fs);
            Console.WriteLine("The model is saved to {0}", ModelPath);
            return trainedModel;
        }
        private static void TestSinglePrediction(MLContext mLContext)
        {
            var areaSample = new AreaCalc()
            {
                floor1 = 123,
                floor2 = 212,
                floor3 = 232,
                floor4 = 12,
                temperature = 23,
                area = 0
            };
            ///
            ITransformer trainedModel;
            using(var stream = new FileStream(ModelPath, FileMode.Open,FileAccess.Read, FileShare.Read))
            {
                trainedModel = mLContext.Model.Load(stream);
            }
            //create prediction engine loaded to the loaded treained model
            var predEngine = trainedModel.CreatePredictionEngine<AreaCalc, AreaCalcPrediction>(mLContext);
            //Score
            var resultprediction = predEngine.Predict(areaSample);
            ///
            Console.WriteLine($"*******************");
            Console.WriteLine($"Predicted Area: {resultprediction.AreaCalc:0.####}, actual fare: 12");
            Console.WriteLine($"*******************");

        }
        private static void PlotRegressionChart(MLContext mLContext,
                                                string testDataSetPath,
                                                int numberOfRecordsToRead,
                                                string[] args)
        {
            ITransformer trainedModel;
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mLContext.Model.Load(stream);
            }
            //Create prediction engine
            var predFunction = trainedModel.CreatePredictionEngine<AreaCalc, AreaCalcPrediction>(mLContext);

            string chartFileName = "";
            using (var pl = new PLStream())
            {
            //USE SVG backend 
            if(args.Length == 1 && args[0] == "svg")
            {
                pl.sdev("svg");
                chartFileName = "AreaCalcRegressionDistribution.svg";
                pl.sfam(chartFileName);
            }
            else
            {
                pl.sdev("pngcairo");
                chartFileName = "AreaCalcRegressionDistribution.png";
                pl.sfam(chartFileName);
            }
            //use white background
            pl.spal0("cmap0_alternate.pal");
            //init plpot
            pl.init();
            //set axis limits
            const int xMinLimit = 0;
            const int xMaxLimit = 35;
            const int yMinLimit = 0;
            const int yMaxLimit = 35;
            pl.env(xMinLimit, xMaxLimit, yMinLimit,yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);
            //set scaling 125%
            pl.schr(0, 1.25);
            //The main title
            pl.lab("Measured", "Predicted", "Distribution of Population");
            pl.col0(1);
            int totalNumber = numberOfRecordsToRead;
            var testData = new AreaCalcReader().GetDataFromCsv(testDataSetPath, totalNumber).ToList();
            //Code symbol to paint
            char code = (char)9;
            pl.col0(2);

            double yTotal = 0;
            double xTotal = 0;
            double xyMultiTotal = 0;
            double xSquareTotal = 0;
            for (int i = 0; i < testData.Count; i++)
            {
                var x = new double[1];
                var y = new double[1];
                //Make Prediction
                var AreaCalcPrediction = predFunction.Predict(testData[i]);

                x[0] = testData[i].AreaCalc;
                y[0] = AreaCalcPrediction.AreaCalc;
                //Paint a dot 
                pl.poin(x,y,code);

                xTotal += x[0];
                yTotal += y[0];

                double multi = x[0] * y[0];
                xyMultiTotal += multi;

                double xSquare = x[0] * x[0];
                xSquareTotal += xSquare;

                double ySquare = y[0] * y[0];

                Console.WriteLine($"-------------------------------------------------");
                Console.WriteLine($"Predicted : {AreaCalcPrediction.AreaCalc}");
                Console.WriteLine($"Actual:    {testData[i].AreaCalc}");
                Console.WriteLine($"-------------------------------------------------");

            }

            double minY = yTotal /totalNumber;
            double minX = xTotal / totalNumber;
            double minXY = xyMultiTotal / totalNumber;
            double minXsquare = xSquareTotal / totalNumber;
            double m = ((minX*minY)-minXY)/ ((minX * minX) - minXsquare);
            double b = minY -(m*minX);
//generic function 
            double x1 = 1;
            double y1 = (m*x1)+b;
            double x2 = 39;
            double y2 = (m*x2)+b;

            var xArray = new double[2];
            var yArray = new double[2];
            xArray[0] = x1;
            yArray[0] = y1;
            xArray[1] = x2;
            yArray[1] = y2;

            pl.col0(4);
            pl.line(xArray, yArray);
            pl.eop();
            pl.gver(out var verText);
            Console.WriteLine("PLplot version " + verText);
            }
            Console.WriteLine("Showing chart...");
            var p = new Process();
            string chartFileNamePath = @".\" + chartFileName;
            p.StartInfo = new ProcessStartInfo(chartFileNamePath)
            {
                UseShellExecute = true
            };
            p.Start();
        }
    }
    public class AreaCalcReader
    {
        public IEnumerable<AreaCalc> GetDataFromCsv(string dataLocation, int numMaxRecords)
        {
            IEnumerable<AreaCalc> records =
                File.ReadAllLines(dataLocation)
                .Skip(1)
                .Select(x => x.Split(','))
                .Select(x => new AreaCalc()
                {
                    floor1 = x[0],
                    floor2 = x[1],
                    floor3 = x[2],
                    floor4 = x[3],
                    temperature = x[4],
                    area = x[5]
                })
                .Take<AreaCalc>(numMaxRecords);

            return records;
        }
    }
}
