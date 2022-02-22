from fastai.basics import *
from fastai.vision.all import *

from tqdm import tqdm
from pathlib import Path
from torchvision.transforms import Normalize

legacy_image_stats = [
    np.array([0.14814416, 0.14217226, 0.13984123]),
    np.array([0.0881476, 0.07823102, 0.07676626]),
]

# faster method
def predict(
    objids, 
    model, 
    image_dir, 
    size=144, 
    image_stats=legacy_image_stats,
    batch_size=64,
):
    """Make predictions given a trained model.

    Parameters
    ----------

    objids : list-like
        List of object names that serve as JPG image filename stems
    image_dir : str or Path
        location of image files
    model : CNN model
        a trained model. Should be initialized with `model.cuda()` if desired
        to run on GPU.
    size : int (144 is default)
        size in pixels of square image
    image_stats : 2-tuple or list of np.array (`legacy_image_stats` is default)
        list-like of means and standard_deviations. np.array must have shape
        corresponding to number of image channels.
    batch_size : int (64 is default)
        Number of objects to process per batch. Depends on GPU memory.

    
    Returns
    -------
    inps : torch.Tensor
        A giant tensor of shape (N, C, size, size), where N is the number of 
        predictions and C is nubmer of channels
    outs : torch.Tensor
        A tensor of predictions of shape (N, N_out), where N_out is the number
        of output neurons (i.e. 1 for single-variable regression, K for 
        classification with K classes, etc).

    Based on https://forums.fast.ai/t/speeding-up-fastai2-inference-and-a-few-things-learned/66179
    """

    type_pipe = Pipeline([PILImage.create])
    item_pipe = Pipeline([Resize(size), ToTensor()])
    normalize = Normalize(*image_stats)
    i2f = IntToFloatTensor()


    # make batches
    batches = []
    batch = []

    k = 0
    for objid in tqdm(df.index, total=len(df.index)):
        im_name = f'{image_dir}/{objid}.jpg'
        batch.append(item_pipe(type_pipe(im_name)))
        k += 1
        if k == batch_size:
            batches.append(torch.cat([normalize(i2f(b))[None] for b in batch]))
            batch = []
            k = 0
            
    # append final batch if not dropped
    if batch != []:
        batches.append(torch.cat([normalize(i2f(b))[None] for b in batch]))
        batch = []
        k = 0

    # predict on batches
    model.eval()

    outs = []
    inps = []

    with torch.no_grad():
        for b in tqdm(batches, total=len(batches)):
            outs.append(model(b))
            inps.append(b)

    inp = torch.cat(inps)
    out = torch.cat(outs)

    return inp, out


# slower method...
def predict_old(filenames, dls, model, saved_model_path):
    """Make classification predictions on test images using trained model.


    Parameters
    ----------

    filenames : list of pathlib.Path objects
        List of filenames of JPG images
    dls : ImageDataLoaders
        Fastai DataLoaders which contains transformations to be applied
    model : CNN model
        An initialized model e.g., `model = xresnet34(n_out=2)`
    saved_model_path : str
        Full path to model that will be opened using `torch.load()`
    
    Returns
    -------
    preds : pd.DataFrame
        A dataframe with index given by filenames, and a column of CNN
        probabilities (corresponding to a positive class prediction for
        a binary classification problem)

    """

    model.load_state_dict(
        torch.load(saved_model_path)
    )
    model.to("cuda")

    test_dl = dls.test_dl(filenames, num_workers=8, bs=bs)

    m = model.eval()
    outputs = []
    with torch.no_grad():
        for (xb,) in tqdm(iter(test_dl), total=len(test_dl)):
        outputs.append(m(xb).cpu())

    outs = torch.cat(outputs)
    outs = outs.softmax(1)

    names = list(x.stem for x in filenames)
    names = np.array(names, dtype=str)

    # assume that index-1 is positive class
    preds = pd.DataFrame({f"p_CNN": outs[:, 1]}, index=names)
    
    return preds

