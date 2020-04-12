# Dense Iris Landmarks
Repo for dense iris landmarks localization with synthesized eye dataset.

## Repo Structure
./config.py: config file 
./loss.py: loss function
./checkpoint.py: save the trained model
./tools: some utitilies
./test: test results

## Prepare Dataset
There are two methods to prepare the training data.
1. You could use the software [here](https://www.cl.cam.ac.uk/research/rainbow/projects/unityeyes/tutorial.html) to synthesiz all kinds of data yourself. Then use scripts in `./gen_dataset` to generate training data.

2. You could also use the dataset I provided. directly. Just download the dataset and put train images in `./data` directory. 
   In this case the annotations are already prepared in `annotations` directory.
   
   Adress:https://pan.baidu.com/s/1gzYAVvEuhuu6L8tos3zXAQ  
   Password:990n
   
## Train
config all the training parameters in `config.py`
RUN `python training/train.py` to train your model.

Training results are kept in `results` directory.

## Test

Put well croped eye images in `./data/test`
RUN `python test/test_image.py` to test your model.

[!result.jpg](./images/epoch_002.jpg)

   

