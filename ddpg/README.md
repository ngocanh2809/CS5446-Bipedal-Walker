# Go through the results:
1. Download the runs result
For bash user, run:
```
cd CS5446-Bipedal-Walker #Get to main directory
mkdir out
cd out
gdown --id 1NgFk_A1HIslTAo6vy3dlomclqB8Sl2zg
unzip ddpg.zip
rm ddpg.zip
```
Or download and unzip the zip files in https://drive.google.com/drive/folders/15X8B8cZyZg4vUaSazgXQClFJZ5B5etzs

In the out/ddpg folder, there are 6 experiment:
- ez_lowerLR: default ddpg trained on easy mode (trained with learning_rate = 0.0001, and succeeded)
- ez_higherLR: default ddpg trained on easy mode (first trained with learning_rate = 0.001, and failed)
- hardcore: default ddpg trained on hardcore mode
- ez_noidle_runfaster: ddpg trained on easy mode and using reward wrapper no_idle and run_faster
- ez_nolegcontact: ddpg trained on easy mode and using reward wrapper no_idle and no_leg_contact
- ez_noidle_runfaster: ddpg trained on easy mode and using reward wrapper no_idle and jump_higher

Within each subfolder, there can be some of these files:
- best_weights: best_model.zip saved by gym wrapper
- log: logs of the training process saved by gym wrapper
- video: videos of the training process saved by gym wrapper
- weights: weights saved every 50,000 steps
- best_on_hardcore: best_model.zip tested on hardcore
- best_on_ez: best_model.zip tested on easy

# Run the code:

1. Install a torch version appropriate to your computer
2. Install packages
```
pip install -r requirements.txt
```

3. To Train
```
python ddpg/ddpg_sb3.py out/path/to/output_folder
```
Add tag --hardcore for hardcore and --record to save video
For example:
```
python ddpg/train_ddpg.py out/ddpg/test --hardcore --record
```

4. To Train with modified reward
```
python -m ddpg.train_ddpg_modreward path/to/outfolder --record --mod reward_modifier1 reward_modifier2 reward_modifier3
```
Add tag --hardcore for hardcore and --record to save video \
Adding strings of reward wrapper string for the wrappers \
Has to be one, some or all of [no_idle, run_faster, jump_higher, no_leg_contact] \

For example:
```
python -m ddpg.train_ddpg_modreward out/ddpg/test  --record --mod no_idle jump_higher
```
4. To Evaluate and Visualize, output to video 
```
python ddpg/eval_ddpg.py path/to/model/excluding_.zip_postfix --mode viz --output output/path
```

5. To Evaluate, output to video 1 model
```
python ddpg/eval_ddpg.py path/to/model/excluding_.zip_postfix --mode eval --output output/path
```
Add tag --hardcore for hardcore and --record to save video 


For example:
```
python ddpg/eval_ddpg.py out/ddpg/ez/weights/best_model --mode eval  --output out/ddpg/ez_lowerLR/best
```

6. For extra evaluation including more details and especially average speed, run eval.py   