# Personalized PageRank Fingerprint Framework for Seedless Graph Matching

## About

This repository refers to the framework developed as part of a master’s dissertation addressing the classic problem of finding node correspondences between two highly similar graphs. The work introduces a novel seedless approach based on Personalized PageRank fingerprints, employing an iterative algorithm that progressively maps nodes across rounds. In each round, the algorithm defines anchor-mapped pairs to enhance fingerprints and then determines matchings in the subsequent round.

## Publications

- [A Personalized PageRank Fingerprint Framework for Seedless Graph Matching](https://www.cos.ufrj.br/uploadfile/publicacao/3178.pdf)  
Master's dissertation presented to COPPE/UFRJ as a partial fulfillment of the requirements for the degree of Master of Science (M.Sc.)

## Getting Started

### Running using uv

Run the framework with [uv](https://github.com/astral-sh/uv), a fast Python package and project manager - no need to manually setup a virtual environment:

```bash
uv run graph-matching-ppr-framework/main.py
```

Here are the general options for the *main.py* file; please also refer to the `PageRankSettings`, `DistanceMetricsSettings` and `SortMethod` options defined in the file.

```text
usage: main.py [-h] -a  -b  [-o ] [-m ] [-v]

options:
  -h, --help           show this help message and exit
  -a, --edgelist-a     edgelist filepath for graph A
  -b, --edgelist-b     edgelist filepath for graph B
  -o, --progress-out   progress output filepath
  -m, --matching-out   matching output filepath
  -v, --verbose        enable verbose output
```

To open JupyterLab (Notebook Interface), run the command below:

```bash
uv run --with jupyter jupyter lab
```

## Differences

When porting the code to this repository, some bugs were fixed and certain equations were updated, which may affect previously reported results. The equations shown below reflect these revisions; for the original formulations and their explanations, please refer to the dissertation.

**Equation 3.5**: Distance between two sequences of PPR values, where the viewpoint and vertex of interest are different. Amortization is $`\sigma^{t-1}`$, no longer $`\sigma^{t-t^*}`$.

```math
    d'(
    R_{u_{1i}}(a_{1k})
    ,
    R_{u_{2j}}(a_{2k})
    )
    =
    \sum_{t=t^*}^T
    \frac{
        \left| R_{u_{1i}}(a_{1i}, t)-R_{u_{2j}}(a_{2j}, t) \right|
    }
    {
        \max \left( 
        R_{u_{1i}}(a_{1k}, t) , R_{u_{2j}}(a_{2k}, t) 
        \right)
    }
    \sigma^{t-1}
```

**Equation 3.7**: New confidence metric $`w(a_{1k}, a_{2k})`$ for anchor pair.

```math
    w
    (
    a_{1k}
    ,
    a_{2k}
    )
    =
    \frac
    {
        \frac{1}
        {
        d
        (
        R_{a_{1k}}
        ,
        R_{a_{2k}}
        )
        + \epsilon
        }
    }
    {
        \sum_{(a_{1x} , a_{2x}) \in S_{h-1}}
        \frac{1}
        {
        d
        (
        R_{a_{1x}}
        ,
        R_{a_{2x}}
        )
        + \epsilon
        }
    }
```
