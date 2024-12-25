using Pattern = (float[] inputs, float label);

#region Load HWD Data
List<Pattern> train_patterns = new();

var file_Path = @"C:\Dev\ANN\HWD.txt";

var digits = new List<float[]>();
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
        if (digits.Count == labels.Count)
        {
            digits.Add(row);
        }
        else
        {
            var lastImage = digits[^1];
            Array.Resize(ref lastImage, lastImage.Length + row.Length);
            Array.Copy(row, 0, lastImage, lastImage.Length - row.Length, row.Length);
            digits[^1] = lastImage;
        }
    }
}

for (int i = 0; i < digits.Count; i++)
{
    Pattern pattern = new();
    pattern.inputs = digits[i];
    pattern.label = labels[i];
    train_patterns.Add(pattern);
}
#endregion

bool stop = false;
int k = 10;
List<Cluster> clusters = new();
Pattern[] initial_center_patterns = train_patterns.OrderBy(x => Guid.NewGuid()).Take(k).ToArray();

for (int i = 0; i < initial_center_patterns.Length; i++)
{
    clusters.Add(new Cluster(initial_center_patterns[i].inputs));
}

foreach (var pattern in train_patterns)
{
    clusters[GetNearestCenter(pattern)].AddMember(pattern);
}

while (!stop)
{
    foreach (Cluster cluster in clusters)
        stop = cluster.UpdateCenter();

    foreach (var pattern in train_patterns)
    {
        foreach (var cluster in clusters)
            cluster.RemoveMember(pattern);

        clusters[GetNearestCenter(pattern)].AddMember(pattern);
    }
}

int GetNearestCenter(Pattern pattern)
{
    float min_distance = float.PositiveInfinity;
    int min_index = 0;

    for (int i = 0; i < clusters.Count; i++)
    {
        float distance = clusters[i].GetDistance(pattern.inputs);
        if (distance < min_distance)
        {
            min_distance = distance;
            min_index = i;
        }
    }

    return min_index;
}

#region Evaluate
float[] classes = new float[10];
for (int i = 0; i < 10; i++)
    classes[i] = float.NaN;

int change_in_cluster = 0;

foreach (var pattern in train_patterns)
{
    int assigned_cluster = 0;

    foreach (var cluster in clusters)
        if (cluster.HasMemeber(pattern)) assigned_cluster = clusters.IndexOf(cluster);

    if (classes[(int)pattern.label] == float.NaN)
        classes[(int)pattern.label] = assigned_cluster;
    else if (classes[(int)pattern.label] != assigned_cluster)
    {
        change_in_cluster++;
        Console.WriteLine($"cluster for label {pattern.label} changed from {classes[(int)pattern.label]} to {assigned_cluster}");
        classes[(int)pattern.label] = assigned_cluster;
    }
}

Console.WriteLine($"total changes in cluster = {change_in_cluster}");
#endregion

class Cluster
{
    private float[] center;
    private List<Pattern> members = new();

    public float[] Center { get { return center; } }
    public List<Pattern> Members { get { return members; } }

    public Cluster(float[] center)
    {
        this.center = center;
    }

    public void AddMember(Pattern pattern)
    {
        members.Add(pattern);
    }

    public void RemoveMember(Pattern pattern)
    {
        if (!HasMemeber(pattern)) return;

        members.Remove(pattern);
    }

    public bool HasMemeber(Pattern pattern)
    {
        return members.Exists(x => x == pattern);
    }

    public float GetDistance(float[] input)
    {
        return MathF.Sqrt(center.Zip(input, (a, b) => MathF.Pow(a - b, 2)).Sum());
    }

    public bool UpdateCenter()
    {
        float[] new_center = CalculateAverage();

        if(GetDistance(new_center) < 0.1f)
            return true;
        else
        {
            center = new_center;
            return false;
        }
    }

    private float[] CalculateAverage()
    {
        List<float[]> inputs_list = new();
        foreach(Pattern pattern in members)
        {
            inputs_list.Add(pattern.inputs);
        }
        int length = inputs_list[0].Length;

        float[] averages = new float[length];
        for (int i = 0; i < length; i++)
        {
            averages[i] = inputs_list.Average(a => a[i]);
        }

        return averages;
    }
}