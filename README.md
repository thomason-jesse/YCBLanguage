# YCBLanguage

This quickly walks you through: 
- extracting processed RGBD features from before / after dropping behavior executions;
- extracting vector representations of RGB, depth, language, and visual information based on RGBD, referring expression, and YCB static image data about objects preprocessed with the above script, GloVe embeddings, and ResNet, respectively; and
- training neural networks that take in those vector representations to predict "in"/"on" affordance labels between pairs of objects.

First, move into `mturk_scripts`

`> cd mturk_scripts`

#### Generating RGBD features.

To extract processed RGBD features from before / after dropping behavior executions:
(TODO: We can play with better feature normalization and processing for RGB and Depth representations; currently, I'm just calculating the raw difference After - Before for RGB and 1 / After - 1 / Before for Depth.)

`> python extract_robot_features.py --splits_infile ../data/3class_object_split.json --features_indir ../src/arm_table_data/ --gt_infile ../src/robo_gt.csv --hand_features_outfile ../data/robo_hand.json --rgbd_features_outfile ../data/robo_rgbd.json`

The "hand features" here are the old, vestigial, hand-engineered radial depth and RGB features that we're no longer using. The `--features_indir` specifies the directory where you've decided to store all the data dumps from Rosario since he mounted the arm on the table, rather than the wheelchair, to get higher quality data. The key output here is `--rgbd_features_outfile` where the RGBD Before - After features are stored for later processing.

#### Generating input/output vectors.

Next, we prep all modalities' vectors using:

`> python prep_torch_data.py --infile ../data/3class_object_split.json --metadata_infile ../data/all_data.json --glove_infile /hdd/jdtho/glove/glove.6B.300d.txt --robot_infile ../data/robo_rgbd.json --gt_infile ../src/robo_gt.csv --rgbd_only 1 --out_fn ../data/torch_ready/rgbd_only_dev`

Lots of arguments! The big takeaway is that `--out_fn` is the prefix where we're going to write that argument plus `.on` and `.in` JSON dumps of feature vectors for each modality: RGB, D, Vision, and Language, plus the target labels for the affordance, for each pair of objects.

More key ideas: `gt_infile` tells the model where the ground truth CSV lives, and uses its labels to overwrite the human annotations in the dev/test fold, then throws up warnings for the unlabeled examples that remain (eventually, we will have annotated them all as a team).

Finally, note here that we're limiting ourselves to `rgbd_only` data: recording features only for the subset of pairs Rosario has collecting RGBD data for using the robot. We can also get information for all pairs, but with RGB and D feature vectors set to None, using:

`> python prep_torch_data.py --infile ../data/3class_object_split.json --metadata_infile ../data/all_data.json --glove_infile /hdd/jdtho/glove/glove.6B.300d.txt --robot_infile ../data/robo_rgbd.json --rgbd_only 0 --out_fn ../data/torch_ready/all_dev`

In both cases, note that I'm naming the output for the `dev` set. If the optional `test` flag is set in these commands, the `test` set will be used instead of `dev`, but we really don't want to do that until our final evaluation, to be safe and not accidentally peak at the test data!

#### Training models.

Okay, finally, now that we have all our vector inputs/outputs prepped, we can run models with:

`> python run_rss_models.py --models glove,resnet,rgbd,mc --input ../data/torch_ready/rgbd_only_dev`

or, for all data but no RGBD model, 

`> python run_rss_models.py --models glove,resnet,mc --input ../data/torch_ready/all_dev`

The `--models` flag tells the script which models to train of `mc` majority class, `glove` language, `resnet` vision, and `rgbd` the conv-based RGBD model.

These architectures are pretty basic and hopefully self-explanatory. They live in `models.py`. The language and vision models are perceptron models with single hidden layers, while the RGBD models are 3 deep conv stacks that process RGB and D separately and concatenate the outputs, running them through a single fully-connected layer and outputting them to a hidden dimension whose size matches that of the hidden layer in language in vision.

#### Next modeling steps.

The next step will be to make an architecture that, instead of predicting affordance labels from these hidden layers directly per modality, concatenates them and does another FC prediction from that multi-modal input to the affordance, an architecture where we can pretrain the L+V models using all data, then tune them on RGBD data.

Finally, things like the `hyperparameter_sweep` file need to be updated to function with the new model running script to sweep over the (smaller) number of architecture choices and then be run over the larger pretrain + tune pipeline, once it's in place.