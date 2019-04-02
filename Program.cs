using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
//using PLplot;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
//using Microsoft.ML.Transforms.Categorical;
//using Microsoft.ML.Transforms.Normalizers;
using static Microsoft.ML.Transforms.NormalizingEstimator;
using KDP.DataStructures;
using Microsoft.Data.DataView;

namespace KDP
{
    class Program
    {
        private static string  AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string  BaseDataSetsLocation = @"/home/y7x/Documents/DEV_APPS_ONGOING/Orzak_dynamics/KDP/DATA";
        private static string  TrainDataPath = $"{BaseDataSetsLocation}/fakeData_train.csv";
        private static string  TestDataPath = $"{BaseDataSetsLocation}/fakeData_test.csv";
        private static string  BaseModelPath = @"MLModel";
        private static string  ModelPath = $"{BaseModelPath}/KinneretMLmodel.zip";



        static void Main(string[] args)
        {
            MLContext mLContext = new MLContext(seed: 0);
            BuildTrainEvaluteAndSaveModel(mLContext);
            TestSinglePrediction(mLContext);
            System.Console.WriteLine(" FINISH ");
            System.Console.ReadLine();
            //Al final
        }
        private static ITransformer BuildTrainEvaluteAndSaveModel(MLContext mLContext)
        {
            TextLoader textLoader1 = mLContext.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("temperature", DataKind.Single, 0),
                new TextLoader.Column("floor1", DataKind.Single, 1),
                new TextLoader.Column("floor2", DataKind.Single, 2),
                new TextLoader.Column("floor3", DataKind.Single, 3),
                new TextLoader.Column("floor4", DataKind.Single, 4),
                new TextLoader.Column("areaEvaluated", DataKind.Single, 5)

            }
            ,hasHeader: true, 
            separatorChar:','
            );
            IDataView baseTrainingDataView = textLoader1.Load(TrainDataPath);
            IDataView testDataView = textLoader1.Load(TestDataPath);
            var cnt = baseTrainingDataView.GetColumn<float>(mLContext, "areaEvaluated").Count();
            IDataView  trainingDataView = mLContext.Data.FilterRowsByColumn(baseTrainingDataView, "areaEvaluated", lowerBound:150, upperBound:350);
            var cnt2 = trainingDataView.GetColumn<float>(mLContext, "areaEvaluated").Count();

            var dataProcessPipeline = mLContext.Transforms.CopyColumns("Label","areaEvaluated")
                                .Append(mLContext.Transforms.Normalize(outputColumnName:"temperature",inputColumnName:"temperature",mode: NormalizerMode.MeanVariance))
                                .Append(mLContext.Transforms.Normalize(outputColumnName:"floor1",inputColumnName:"floor1",mode: NormalizerMode.MeanVariance))
                                .Append(mLContext.Transforms.Normalize(outputColumnName:"floor2",inputColumnName:"floor2",mode: NormalizerMode.MeanVariance))
                                .Append(mLContext.Transforms.Normalize(outputColumnName:"floor3",inputColumnName:"floor3",mode: NormalizerMode.MeanVariance))
                                .Append(mLContext.Transforms.Normalize(outputColumnName:"floor4",inputColumnName:"floor4",mode: NormalizerMode.MeanVariance))
                                .Append(mLContext.Transforms.Concatenate("Features","temperature", "floor1", "floor2", "floor3","floor4")
                                );

           var trainer = mLContext.Regression.Trainers.StochasticDualCoordinateAscent(labelColumnName:"Label", featureColumnName:"Features");
           var trainingPipeline = dataProcessPipeline.Append(trainer);

           System.Console.WriteLine("training the model");
           var trainedModel = trainingPipeline.Fit(trainingDataView);

           System.Console.WriteLine("Testing model");
           IDataView predictions = trainedModel.Transform(testDataView);
           var metrics = mLContext.Regression.Evaluate(predictions, label:"Label", score:"Score");

           using (var fs = File.Create(ModelPath))
                trainedModel.SaveTo(mLContext, fs);

            System.Console.WriteLine("the model is saved to {0}", ModelPath);
            return trainedModel;

                                
        }
        private static void TestSinglePrediction(MLContext mLContext)
        {
            var areaCalcSample = new AreaCalc()
            {
                //21,70,58,55,59,167
                temperature = 21,
                floor1 = 70,
                floor2 = 58,
                floor3 = 55,
                floor4 = 59,
                areaEvaluated = 0
            };
            ITransformer trainedModel;
            using (var stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                trainedModel = mLContext.Model.Load(stream);
            }

            var predEngine = trainedModel.CreatePredictionEngine<AreaCalc, AreaCalcPrediction>(mLContext);

            var resultprediction = predEngine.Predict(areaCalcSample);
            System.Console.WriteLine("+++++++++++++++++++++++++++++++++++");
            
            System.Console.WriteLine($"+++++Predicted Area population amount {resultprediction.AreaCalcAmount:0}+++ Actual fare: ? +++++++++++++++++++++++++++");
            
            System.Console.WriteLine("+++++++++++++++++++++++++++++++++++");
        }

        public class AreaCalcCsvReader
        {
            public IEnumerable <AreaCalc> GetDataFromCsv(string dataLocation, int numMaxRecords)
            {
                IEnumerable <AreaCalc> records = 
                    File.ReadAllLines(dataLocation)
                    .Skip(1)
                    .Select(x => x.Split(','))
                    .Select(x => new AreaCalc()
                    {
                        temperature = float.Parse(x[0]),
                        floor1 = float.Parse(x[1]),
                        floor2 = float.Parse(x[2]),
                        floor3 = float.Parse(x[3]),
                        floor4 = float.Parse(x[4])
                    })
                    .Take<AreaCalc>(numMaxRecords);
                return records;
            }
        }
    }
}
