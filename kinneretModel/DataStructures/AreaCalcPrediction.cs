using Microsoft.ML.Data;

namespace Regression_AreaCalcPredictions.DataStructures
{
    public class AreaCalcPrediction 
    {
        [ColumnName("Score")]
        public float AreaCalc;
    }
}