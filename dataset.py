"""
HICODet dataset under PyTorch framework

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import json
import numpy as np

from typing import Optional, List, Callable, Tuple

"""
Dataset base classes

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

import os
import pickle
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Callable, List, Optional, Tuple

__all__ = ['DataDict', 'ImageDataset', 'DataSubset', 'DatasetConcat']

class DataDict(dict):
    r"""
    Data dictionary class. This is a class based on python dict, with
    augmented utility for loading and saving
    
    Arguments:
        input_dict(dict, optional): A Python dictionary
        kwargs: Keyworded arguments to be stored in the dict

    Example:

        >>> from pocket.data import DataDict
        >>> person = DataDict()
        >>> person.is_empty()
        True
        >>> person.age = 15
        >>> person.sex = 'male'
        >>> person.save('./person.pkl', 'w')
    """
    def __init__(self, input_dict: Optional[dict] = None, **kwargs) -> None:
        data_dict = dict() if input_dict is None else input_dict
        data_dict.update(kwargs)
        super(DataDict, self).__init__(**data_dict)

    def __getattr__(self, name: str) -> Any:
        """Get attribute"""
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute"""
        self[name] = value

    def save(self, path: str, mode: str = 'wb', **kwargs) -> None:
        """Save the dict into a pickle file"""
        with open(path, mode) as f:
            pickle.dump(self.copy(), f, **kwargs)

    def load(self, path: str, mode: str = 'rb', **kwargs) -> None:
        """Load a dict or DataDict from pickle file"""
        with open(path, mode) as f:
            data_dict = pickle.load(f, **kwargs)
        for name in data_dict:
            self[name] = data_dict[name]

    def is_empty(self) -> bool:
        return not bool(len(self))

class StandardTransform:
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py"""

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, inputs: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)

class ImageDataset(Dataset):
    """
    Base class for image dataset

    Arguments:
        root(str): Root directory where images are downloaded to
        transform(callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version
    """
    def __init__(self, root: str, transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        self._root = root
        self._transform = transform
        self._target_transform = target_transform
        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, i):
        raise NotImplementedError
    
    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tRoot path: {}\n'.format(self._root)
        return reprstr

    def load_image(self, path: str) -> Image: 
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')

class DataSubset(Dataset):
    """
    A subset of data with access to all attributes of original dataset

    Arguments:
        dataset(Dataset): Original dataset
        pool(List[int]): The pool of indices for the subset
    """
    def __init__(self, dataset: Dataset, pool: List[int]) -> None:
        self.dataset = dataset
        self.pool = pool
    def __len__(self) -> int:
        return len(self.pool)
    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.pool[idx]]
    def __getattr__(self, key: str) -> Any:
        if hasattr(self.dataset, key):
            return getattr(self.dataset, key)
        else:
            raise AttributeError("Given dataset has no attribute \'{}\'".format(key))

class DatasetConcat(Dataset):
    """Combine multiple datasets into one

    Parameters:
    -----------
    args: List[Dataset]
        A list of datasets to be concatented
    """
    def __init__(self, *args: Dataset) -> None:
        self.datasets = args
        self.lengths = [len(dataset) for dataset in args]
    def __len__(self) -> int:
        return sum(self.lengths)
    def __getitem__(self, idx: int) -> Any:
        dataset_idx, intra_idx = self.compute_intra_idx(idx, self.lengths)
        return self.datasets[dataset_idx][intra_idx]
    @staticmethod
    def compute_intra_idx(idx: int, groups: List[int]) -> Tuple[int, int]:
        """Assume a sequence has been divided into multiple groups. Given
        a global index and the number of items each group has, find the
        corresponding group index and the intra index within the group

        Parameters:
        -----------
        idx: int
            Global index
        groups: List[int]
            Number of items in each group
        
        Returns:
        --------
        group_idx: int
            Index of the group
        intra_idx: int
            Intra index within the group
        """
        if idx >= sum(groups):
            raise ValueError(
                "The global index should be smaller "
                "than the length of the sequence."
            )
        groups = np.asarray(groups)
        cumsum = groups.cumsum()
        group_idx = np.where(idx < cumsum)[0].min()
        cumsum = np.concatenate([np.zeros(1, dtype=int), cumsum])
        intra_idx = idx - cumsum[group_idx]
        return group_idx, intra_idx

class HICODetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)
    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]
    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]
    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno
    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno
    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno


class HICODetVLMDataset(Dataset):
    def __init__(self, hicodet, preprocess_fn):
        self.hicodet = hicodet
        self.preprocess = preprocess_fn

    def __len__(self):
        return len(self.hicodet)

    def __getitem__(self, idx):
        sample = self.hicodet[idx]  # (image, target)
        return self.preprocess(sample)
    
class HICODet(ImageDataset):
    """
    Arguments:
        root(str): Root directory where images are downloaded to
        anno_file(str): Path to json annotation file
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample 
            and its target as entry and returns a transformed version.
    """
    def __init__(self, root: str, anno_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None) -> None:
        super(HICODet, self).__init__(root, transform, target_transform, transforms)
        with open(anno_file, 'r') as f:
            anno = json.load(f)

        self.num_object_cls = 80
        self.num_interation_cls = 600
        self.num_action_cls = 117
        self._anno_file = anno_file

        # Load annotations
        self._load_annotation_and_metadata(anno)

    def __len__(self) -> int:
        """Return the number of images"""
        # return 32
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image
        
        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
        """
        intra_idx = self._idx[i]

        # for debugging
        image, target = self._transforms(
            self.load_image(os.path.join(self._root, self._filenames[intra_idx])), 
            self._anno[intra_idx]
            )
        target['filename'] = self._filenames[intra_idx]

        return image, target

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(root=' + repr(self._root)
        reprstr += ', anno_file='
        reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        reprstr += '\tImage directory: {}\n'.format(self._root)
        reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        HICODet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type
        
        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def objects(self) -> List[str]:
        """
        Object names 

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i] 
            for _, i, j in self._class_corr]

    @property
    def rare(self) -> List[int]:
        """
        List of rare class indices
        
        Returns:
            list[int]
        """
        return self._rare

    @property
    def non_rare(self) -> List [int]:
        """
        List of non-rare class indices

        Returns:
            list[int]
        """
        return self._non_rare

    def split(self, ratio: float) -> Tuple[HICODetSubset, HICODetSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return HICODetSubset(self, perm[:n]), HICODetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(dict): Dictionary loaded from {anno_file}.json
        """
        idx = list(range(len(f['filenames'])))
        for empty_idx in f['empty']:
            idx.remove(empty_idx)

        num_anno = [0 for _ in range(self.num_interation_cls)]
        for anno in f['annotation']:
            for hoi in anno['hoi']:
                num_anno[hoi] += 1

        self._idx = idx
        self._num_anno = num_anno

        self._anno = f['annotation']
        self._filenames = f['filenames']
        self._image_sizes = f['size']
        self._class_corr = f['correspondence']
        self._empty_idx = f['empty']
        self._objects = f['objects']
        self._verbs = f['verbs']
        self._rare = f['rare']
        self._non_rare = f['non_rare']
