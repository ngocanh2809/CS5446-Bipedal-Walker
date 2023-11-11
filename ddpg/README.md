To install:

1. Install a torch version appropriate to your computer
2. Install packages
```
pip install -r requirements.txt
```

3. To Train
```
python ddpg/ddpg_sb3.py
```

4. To Visualize, output to video 1 model
```
python ddpg/eval_ddpg.py path/to/model --mode viz --hardcore whether-to-hardcore-bool --output output/path
```

5. To Evaluate, output to video 1 model
```
python ddpg/eval_ddpg.py path/to/model/excluding_.zip_postfix --mode eval --hardcore whether-to-hardcore-bool --output output/path
```

For example:
python ddpg/eval_ddpg.py out/ddpg/ez/weights/best_model --mode eval  --output out/ddpg/ez_lowerLR/best
