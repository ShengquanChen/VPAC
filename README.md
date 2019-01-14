# VPAC

# Requirements
- python3 (is preferable)
- numpy
- scipy
- sklearn
- tqdm (only when the VERBOSE is True)

# Usage instructions
Download VPAC.
```
git clone https://github.com/ShengquanChen/VPAC
```
Load in the data which should be arranged as `n_features` by `n_samples`. Fit the model with parameter `latent_dim` specifying the number of latent dimensions, and `n_components` the number of mixture components.
```
from vpac import VPAC
vpac = VPAC(y = data, latent_dim = 5, n_components = 3)
vpac.fit()
```
Predict posterior probability of each component given the data.
```
prob = vpac.predict_proba()
```

# License
This project is licensed under the MIT License - see the LICENSE file for details.