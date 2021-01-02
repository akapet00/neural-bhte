# neural-bhte

The code base for the graduate course in [The Interaction of the Electromagnetic Field and Human Body](https://nastava.fesb.unist.hr/nastava/predmeti/11661) seminar.

The paper titled [Numerical Solution and Uncertainty Quantification of Bioheat Transfer Equation Using Neural Network Approach](https://github.com/antelk/1d-bioheat-transfer/blob/master/preprint/Numerical%20Solution%20and%20Uncertainty%20Quantification%20of%20Bioheat%20Transfer%20Equation%20Using%20Neural%20Network%20Approach.pdf) is based on the seminar and is available on IEEE Xplore: https://ieeexplore.ieee.org/document/9243733

## Cite
```tex
@inProceedings{Kapetanovic2020,
    author={A. L. {Kapetanović} and A. {Šušnjara} and D. {Poljak}},
    booktitle={2020 5th International Conference on Smart and Sustainable Technologies (SpliTech)},
    title={Numerical Solution and Uncertainty Quantification of Bioheat Transfer Equation Using Neural Network Approach},
    year={2020},
    pages={1-6},
    doi={10.23919/SpliTech49282.2020.9243733}}
```

## Run

Recommended OS is Linux or, for Windows 10, Windows Subsystem for Linux (WSL). However, it is important to note that GPU computing is not supported using WSL.

Use `Jupyter Notebook` to run the experiments.

After downloading the repository as follows:
```bash
$ git pull git@github.com:antelk/neural-bhte.git
```
change directory into `neural-bhte` and run:
```bash
$ jupyter notebook
```
Prerequisities are available in `environment.yml`.
It is recommended to create `conda` environment as follows:
```bash
$ conda env create -n neural-bhte -f environment.yml
```
The entire relevant code is in `neural-bhte-implementation.ipynb` notebook.


## License 

[MIT](https://github.com/antelk/covid-19/blob/master/LICENSE)