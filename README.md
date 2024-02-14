# [Clifford Group Equivariant Simplicial Message Passing Networks (ICLR 2024)](https://openreview.net/group?id=ICLR.cc/2024/Conference/Authors&referrer=%5BHomepage%5D(%2F)) 
![CSMPNs](CSMPNs.png "CSMPNs")

**Authors**: Cong Liu*, David Ruhe*, Floor Eijkelboom, Patrick Forré

* [Paper Link](https://openreview.net/group?id=ICLR.cc/2024/Conference/Authors&referrer=%5BHomepage%5D(%2F))

## Abstract
We introduce Clifford Group Equivariant Simplicial Message Passing Networks, a method for steerable E(n)-equivariant message passing on simplicial complexes. Our method integrates the expressivity of Clifford group-equivariant layers with simplicial message passing, which is topologically more intricate than regular graph message passing. Clifford algebras include higher-order objects such as bivectors and trivectors, which express geometric features (e.g., areas, volumes) derived from vectors. Using this knowledge, we represent simplex features through geometric products of their vertices. To achieve efficient simplicial message passing, we share the parameters of the message network across different dimensions. Additionally, we restrict the final message to an aggregation of the incoming messages from different dimensions, leading to what we term shared simplicial message passing. Experimental results show that our method is able to outperform both equivariant and simplicial graph neural networks on a variety of geometric tasks.

## Requirement and Installation
* Requirement: See `requirement.yml`
* Installation: `conda env create -f environment.yml`

## Code Organization
* `csmpns/`: contains the core code snippets.
  * `algebra/`: contains the Clifford Algebra implementation.
  * `configs/`: contains the configuration files. 
  * `data/`: contains necessary (simplicial) data modules.
  * `models/`: contains model and layer implementations.
* `engineer/`: contains the training and evaluation scripts.

## Usage and Datasets
This implementation uses conda environment, change the path of `miniconda/` in `activate.sh` to your local `miniconda/` path and run `sh activate.sh`.

Run `pip install -e .`

### Convex Hull Volume Prediction:  
`sweep_local csmpn/configs/hulls.yaml` 

### Human Walking Motion Prediction:
`sweep_local csmpn/configs/motion.yaml`

### MD17 Atomic Motion Prediction:
`sweep_local csmpn/configs/md17.yaml`

### NBA Players trajectory Prediction:
`sweep_local csmpn/configs/nba.yaml`

## Citation:
If you found this code useful, please cite our paper:

