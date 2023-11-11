To install:

1. Install a torch version appropriate to your computer
2. Install packages
```
pip install -r requirements.txt
```

3. To Train
```
python genetic_alg/evolved_nn.py
```

4. To Visualize, output to video 1 model
```
python genetic_alg/eval_alg.py path/to/model --mode viz --hardcore whether-to-hardcore-bool --output output/path
```

5. To Evaluate, output to video 1 model
```
python genetic_alg/eval_alg.py path/to/model/excluding_.zip_postfix --mode eval --hardcore whether-to-hardcore-bool --output output/path
```

For example:
python genetic_alg/eval_alg.py out/genetic_alg/evolved_nn/weights_ez/gen_39_142.31.pkl --mode eval  --output out/genetic_alg/evolved_nn/weights_ez/eval_on_ez
