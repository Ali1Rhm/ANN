using SOM_Base;

List<(float[] inputs, float label)> train_patterns = new();

SOM som = new(train_patterns);
som.Start();