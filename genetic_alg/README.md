# Go through the results:
1. Download the runs result
For bash user, run:
```
cd CS5446-Bipedal-Walker #Get to main directory
mkdir out
cd out
gdown --id 1xl2oFmDEYVpjg3uLyluFIaZrTaiswbeV
unzip genetic_alg.zip
rm genetic_alg.zip
```

Or download and unzip the zip files in https://drive.google.com/drive/folders/15X8B8cZyZg4vUaSazgXQClFJZ5B5etzs

In the out/genetic_alg/evolved_nn folder, there are 2 subfolders, weights_ez and weights_hardcore
- weights_ez: algorithm trained on ez mode, on 40 generations, population size 48
- weights_hardcore: algorithm trained on hardcore mode, on 60 generations, population size 48

Inside there's a subfolder eval containing the metric_eval.json and png, containing the evaluation of the of the run. Also the video recordings of the inference of the last weight file.

# Run the code:

1. Install a torch version appropriate to your computer
2. Install packages
```
pip install -r requirements.txt
```

From the main directory

3. To Train
```
python genetic_alg/evolved_nn.py path/to/outfolder --gen number-of-generations-to-train --pop_size number-of-agents-per-generation
```
Add tag --hardcore for hardcore mode

For example:
```
python genetic_alg/evolved_nn.py out/test --gen 40 --pop_size 48
```


4. To Visualize and evaluate, output to videos
```
python genetic_alg/eval_alg.py path/to/model --mode viz --output output/path
```

5. To Evaluate, watch the videos as it runs
```
python genetic_alg/eval_alg.py path/to/model --mode eval --output output/path
```
Add tag --hardcore for hardcore mode

For example:
```
python genetic_alg/eval_alg.py out/genetic_alg/evolved_nn/weights_ez/gen_39_142.31.pkl --mode eval  --output out/genetic_alg/evolved_nn/weights_ez/eval_on_ez
```

6. To draw the graph of rewards across generations, run
```
python genetic_alg/reward_graph.py path/to/folder/
```

For example:
```
python genetic_alg/reward_graph.py out/genetic_alg/evolved_nn/weights_ez/
```
