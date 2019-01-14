# PLAsTiCC-Astronomical-Classification-RNN-Search

Kaggle submission. 

## Highlights

Some interesting notes about this approach:

### Focal Loss

The use of this loss function comes from computer vision, particular object detection. The idea is that neural networks can often generalize too much and fail to understand harder examples of data. I think this is perfectly applicable for time series problems too. For more discussion on Focal Loss see [Lazy Neural Networks](https://medium.com/@glenn.kroegel/lazy-neural-networks-8b0de07fd578)

### Cosine Similarity Between Object Encodings

The architecture returns an encoding of each object as well as classification predictions. Encodings can be compared by using a similarity algorithm such as L1 distance or cosine similarity. 'nmslib' is used to do a batch query and return 'k' most similar objects and corresponding scores. This is useful in it's own right to find similar objects. Strange objects would have a relatively low score amongs it's 'k' best. 

See 'find_unknown_objects.ipynb' for calculation.

The model can be toggled to return encodings by setting 'return_state=True' in the forward call.

### Model flexibility

The model is broken into several modules that are all interchangable. For example, the encoding used to do the cosine similarity can easily be switch out to something more expressive. The RNN blocks taking the passband inputs can equally be changed easily. 

## Usage

1. Generate dataloaders by running the notebooks 'preprocessing.ipynb' and 'preprocessing_test_set.ipynb'. The notebooks assume the data files are in the repositories root directory (not there by default).

2. Execute 'train.py'. The model will train for 200 epochs by default. This can be modified in 'config.py'.

3. Run the 'find_unknown_objects.ipynb' notebook to find the most dissimilar objects.

4. TODO: softmax outputs for classification - utilizing the determined unknown objects from (3).