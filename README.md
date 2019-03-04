# YCBLanguage

Source for IROS 2019 submission 143:

"Improving Robot Success Detection using Static Object Data." Rosario Scalise, Jesse Thomason, Yonatan Bisk, and Siddhartha Srinivasa

### Supplementary Material

### Rosbags

Sensor streams and robot trajectory rosbag files are available on request. These files total around 100GB.

### Success Detection Models

This README walks you through: 
- extracting processed RGBD features from before / after dropping behavior executions;
- extracting vector representations of RGB, depth, language, and visual information based on RGBD, referring expression, and YCB static image data about objects; and
- training and evaluating end-to-end networks that take in those vector representations to classify "in"/"on" success labels between pairs of objects.

These models are implemented with the `pytorch` library. To download this and other requirements, run:

`pip install -r requirements.txt`

#### Downloading Data.

Static images (with preprocessed ResNet features already extracted) and downsampled RGB-D trial data can be downloaded at:

[Static YCB Object Images](https://drive.google.com/open?id=1agZgomkPywxQCp91Usx1gqEdfpIMCr0L)

Move this `imgs` directory to `data/imgs/`.

[RGB-D trial data](https://drive.google.com/open?id=1kK2Zj7NZ_IDtSO0MlbIs2ux1Vx0vKh3i)

You'll need to place the RGB-D trial data in its own directory with no other JSON files.

Additionally, you will need a local copy of GloVe embedding vectors.

[GloVe Vectors](http://nlp.stanford.edu/data/glove.6B.zip)

For our experiments, we used `glove.6B.300d.txt`, 6 billion tokens of coverage with 300 dimensional embeddings.


#### Generating RGBD features.

First, move into `scripts`

`cd scripts`

To extract processed RGBD features from pre- and post-manipulation scans:

`python extract_robot_features.py --splits_infile ../data/3class_object_split.json --features_indir [your path to RGB-D trial data directory] --rgbd_features_outfile [RGB-D features json]`

The key output here is `--rgbd_features_outfile` where the RGB-D features are stored for later processing.

#### Generating input/output training and evaluation data.

Next, prep all modalities' vectors using:

`python prep_torch_data.py --infile ../data/3class_object_split.json --metadata_infile ../data/all_data.json --glove_infile [your GloVe embeddings txt] --robot_infile [RGB-D features json] --exec_robo_indir ../src/robo_exec_gt/ --exec_human_indir ../src/human_exec_gt/ --rgbd_only 1 --out_fn [path for vectors]/rgbd_only_dev`

`--out_fn` is the prefix for `.on` and `.in` JSON dumps of feature vectors for each modality: RGB, D, Vision, and Language, plus the target labels for the task, for each pair of objects. Labels are calculated for mturk, robo, and human execution, where applicable.

Note here that we're limiting the output to `rgbd_only` data: recording features only for the subset of Robot Pairs that have RGB-D trial data for from the robot. We can also get information for All Pairs, but with RGB and D feature vectors set to None, using:

`python prep_torch_data.py --infile ../data/3class_object_split.json --metadata_infile ../data/all_data.json --glove_infile [your GloVe embeddings txt] --robot_infile [RGB-D features json] --exec_robo_indir ../src/robo_exec_gt/ --exec_human_indir ../src/human_exec_gt/ --rgbd_only 0 --out_fn [path for vectors]/all_dev`

If the optional `test` flag is set in these commands, the `test` set will be used instead of `dev`, e.g.,

`python prep_torch_data.py --infile ../data/3class_object_split.json --metadata_infile ../data/all_data.json --glove_infile [your GloVe embeddings txt] --robot_infile [RGB-D features json] --exec_robo_indir ../src/robo_exec_gt/ --exec_human_indir ../src/human_exec_gt/ --rgbd_only 1 --test 1 --out_fn [path for vectors]/rgbd_only_test`

`python prep_torch_data.py --infile ../data/3class_object_split.json --metadata_infile ../data/all_data.json --glove_infile [your GloVe embeddings txt] --robot_infile [RGB-D features json] --exec_robo_indir ../src/robo_exec_gt/ --exec_human_indir ../src/human_exec_gt/ --rgbd_only 0 --test 1 --out_fn [path for vectors]/all_test`

#### Training and evaluating models.

Run the majority class baseline, egocentric, and egocentric+object data models with:

`python train_and_eval_models.py --models mc,rgbd,rgbd+glove+resnet --round_m_to_n 1 --outdir [robot_pairs_trained_models] --input [path for vectors]/rgbd_only_dev --train_objective robo --test_objective robo --random_restarts 1,2,3,4,5,6,7,8,9,10`

Note that this loads and evaluates on the `dev` fold, and uses `robo` target labels from robot trials.

All results in our submission use 10 random restarts with the given seeds. Omit this flag to run a single instance of the given models with a random seed.

To run the egocentric+(pretrained) object data model, first pretrain the object data layers on All Pairs with target labels from mturk annotations:

`python train_and_eval_models.py --models glove+resnet --round_m_to_n 1 --outdir [all_pairs_trained_models] --input [path for vectors]/all_dev --train_objective mturk --test_objective mturk --random_restarts 1,2,3,4,5,6,7,8,9,10`

Then load the pretrained weights for training and evaluating against robo labels:

`python train_and_eval_models.py --models rgbd+glove+resnet --round_m_to_n 1 --outdir [robot_pairs_trained_models] --input [path for vectors]/rgbd_only_dev --train_objective robo --test_objective robo ----lv_pretrained_dir [all_pairs_trained_models] --random_restarts 1,2,3,4,5,6,7,8,9,10`

This trains a separate model for each set of weights found in the `[all_pairs_trained_models]` directory.

#### Viewing contrastive examples.

To generate a summary of pairs for which two trained models disagree, record their predictions during training and evaluation with the optional `predictions_outfile` flag, e.g.,

`python train_and_eval_models.py --models rgbd,rgbd+glove+resnet --round_m_to_n 1 --outdir [robot_pairs_trained_models] --input [path for vectors]/rgbd_only_dev --train_objective robo --test_objective robo ----lv_pretrained_dir [all_pairs_trained_models] --random_restarts 1,2,3,4,5,6,7,8,9,10 --predictions_outfile [dev predictions json]`

Subsequently, use `view_contrastive_pairs.py` to generate the summary:

`cd analysis`
`python view_contrastive_pairs.py --predictions_infile [dev predictions json] --metadata_infile ../../data/all_data.json --torch_infile [path for vectors]/rgbd_only_dev --use_highest_acc --splits_infile ../../data/3class_object_split.json`
