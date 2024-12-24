using SOM_Base;

List <(float[] inputs, float label)> train_patterns = new();

var file_Path = @"C:\Dev\ANN\HWD.txt";

var images = new List<float[]>();
var labels = new List<float>();

foreach (var line in File.ReadLines(file_Path))
{
    if (string.IsNullOrWhiteSpace(line)) continue;

    if (line.Trim().Length == 1 && int.TryParse(line, out int label))
    {
        labels.Add(label);
    }
    else
    {
        var row = Array.ConvertAll(line.Trim().ToCharArray(), c => c == '1' ? 1.0f : 0.0f);
        if (images.Count == labels.Count)
        {
            images.Add(row);
        }
        else
        {
            var lastImage = images[^1];
            Array.Resize(ref lastImage, lastImage.Length + row.Length);
            Array.Copy(row, 0, lastImage, lastImage.Length - row.Length, row.Length);
            images[^1] = lastImage;
        }
    }
}

for(int i = 0; i < images.Count; i++)
{
    train_patterns.Add((images[i], labels[i]));
}

Console.WriteLine("Linear Topology:");
SOM som = new(train_patterns, 1024, 10, 0.6f, 10 ,200, "linear");
som.Start();
Evaluate(som, 10);

Console.WriteLine("\nSquare Topology:");
SOM som_square = new(train_patterns, 1024, 10, 0.6f, 10, 200, "square");
som_square.Start();
Evaluate(som_square,10);

Console.WriteLine("\nHexagon Topology:");
SOM som_hexagon = new(train_patterns, 1024, 10, 0.6f, 10, 200, "hexagon");
som_hexagon.Start();
Evaluate(som_hexagon, 10);


void Evaluate(SOM som, int labels_count)
{
    float[] labels = new float[labels_count];
    for (int i = 0; i < labels_count; i++)
        labels[i] = float.NaN;

    int change_in_cluster = 0;

    foreach (var pattern in train_patterns)
    {
        int assigned_cluster = som.GetBestMatchingUnit(pattern.inputs);

        if (labels[(int)pattern.label] == float.NaN)
            labels[(int)pattern.label] = assigned_cluster;
        else if (labels[(int)pattern.label] != assigned_cluster)
        {
            change_in_cluster++;
            Console.WriteLine($"cluster for label {pattern.label} changed from {labels[(int)pattern.label]} to {assigned_cluster}");
            labels[(int)pattern.label] = assigned_cluster;
        }
    }

    Console.WriteLine($"total changes in cluster = {change_in_cluster}");
}