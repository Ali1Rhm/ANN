using MLP_Base;
using OfficeOpenXml;

ExcelPackage.LicenseContext = LicenseContext.NonCommercial;

string featuresPath = @"C:\Dev\ANN\thyroidInputs.xlsx";
string labelsPath = @"C:\Dev\ANN\thyroidTargets.xlsx";
List<(float[] input, int[] label)> train_patterns = new();

// Load features
using (var featurePackage = new ExcelPackage(new FileInfo(featuresPath)))
using (var labelPackage = new ExcelPackage(new FileInfo(labelsPath)))
{
    ExcelWorksheet featureSheet = featurePackage.Workbook.Worksheets[0];
    ExcelWorksheet labelSheet = labelPackage.Workbook.Worksheets[0];

    int rows = featureSheet.Dimension.Rows;
    int cols = featureSheet.Dimension.Columns;

    List<int[]> labels = new();
    for (int i = 1; i <= cols; i++)
    {
        labels.Add([
            Convert.ToInt32(labelSheet.Cells[1, i].Value) == 0 ? -1 : 1,
            Convert.ToInt32(labelSheet.Cells[2, i].Value) == 0 ? -1 : 1,
            Convert.ToInt32(labelSheet.Cells[3, i].Value) == 0 ? -1 : 1,
        ]);
    }

    for (int i = 1; i <= cols; i++)
    {
        float[] inputs = new float[rows];
        for (int j = 1; j <= rows; j++)
        {
            inputs[j - 1] = Convert.ToSingle(featureSheet.Cells[j, i].Value);
        }

        train_patterns.Add((inputs, labels[i - 1]));
    }
}

//List<(float[] input, int[] label)> test_patterns = new();

/*for (int i = 0; i < train_patterns.Count * 0.1; i++)
{
    Random random = new Random();
    int randomIndex = random.Next(train_patterns.Count);
    var element = train_patterns[randomIndex];
    train_patterns.RemoveAt(randomIndex);
    test_patterns.Add(element);
}*/

int input_size = 21;
int[] layers_size = { 21, 3 };
float alpha = 0.01f;
int max_epoch = 100;

MLP mlp = new(input_size, layers_size, alpha, max_epoch, train_patterns);
mlp.TrainMLP();

#region Test

string[] classNames = { "Subnormal functioning", "Hyperfunction", "Normal" };
int[] actualLabels = new int[train_patterns.Count];
int[] predictedLabels = new int[train_patterns.Count];
int index = 0;

foreach ((var input, int[] label) in train_patterns)
{
    int label_index = label[0] == 1 ? 0 : (label[1] == 1 ? 1 : 2);

    var predictions = mlp.TestMLP(input);
    float max = Math.Max(predictions[0], Math.Max(predictions[1], predictions[2]));
    int prediction_index = predictions.IndexOf(max);

    if(label_index == 1)
    {
        WriteLine($"{predictions[0]}, {predictions[1]}, {predictions[2]}");
    }

    actualLabels[index] = label_index;
    predictedLabels[index] = prediction_index;
    index++;
}

int numClasses = classNames.Length;

int[,] confusionMatrix = new int[numClasses, numClasses];

for (int i = 0; i < actualLabels.Length; i++)
{
    confusionMatrix[actualLabels[i], predictedLabels[i]]++;
}

WriteLine("Confusion Matrix:");
Write("\t\t");
foreach (var className in classNames)
{
    Write(className.PadRight(20) + "\t");
}
WriteLine();
for (int i = 0; i < numClasses; i++)
{
    Write(classNames[i].PadRight(20) + "\t");
    for (int j = 0; j < numClasses; j++)
    {
        Write(confusionMatrix[i, j].ToString().PadRight(20) + "\t");
    }
    WriteLine();
}

WriteLine("\nMetrics:");
double overallAccuracy = 0;
for (int i = 0; i < numClasses; i++)
{
    int truePositive = confusionMatrix[i, i];
    int falsePositive = 0, falseNegative = 0, total = 0;

    for (int j = 0; j < numClasses; j++)
    {
        if (i != j) falsePositive += confusionMatrix[j, i];
        if (i != j) falseNegative += confusionMatrix[i, j];
        total += confusionMatrix[i, j];
    }

    double precision = truePositive / (double)(truePositive + falsePositive);
    double recall = truePositive / (double)(truePositive + falseNegative);
    double f1Score = 2 * (precision * recall) / (precision + recall);

    WriteLine($"{classNames[i]}:");
    WriteLine($"  Precision: {precision:F2}");
    WriteLine($"  Recall: {recall:F2}");
    WriteLine($"  F1-Score: {f1Score:F2}");

    overallAccuracy += truePositive;
}

overallAccuracy /= actualLabels.Length;
WriteLine($"\nOverall Accuracy: {overallAccuracy:P2}");

#endregion